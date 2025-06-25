# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Functions to convert JSON files to Python DSP pipelines."""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Annotated, Any, Optional, Union, TypeVar, TypeAlias

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.stages as Stages
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.design.stage import StageOutputList, all_stages
from audio_dsp.models.signal_chain import Fork
from audio_dsp.models.stage import all_models, StageModel
import json
import re
import warnings

BAD_NAMES = []

# Define the union type alias for all stage models
_stage_Models = Annotated[
    Union[
        tuple((SkipJsonSchema[i] if i.__name__ in BAD_NAMES else i) for i in all_models().values())
    ],
    Field(discriminator="op_type"),
]


class Input(BaseModel, extra="ignore"):
    """Pydantic model of the inputs to a DSP graph."""

    name: str = Field(..., description="Name of the input")
    channels: int
    output: list[tuple[str, int]] = Field(
        default_factory=list,
        description="List of output edges as (node_name, index) tuples",
    )


class Output(BaseModel, extra="ignore"):
    """Pydantic model of the outputs of a DSP graph."""

    name: str = Field(..., description="Name of the output")
    input: list[tuple[str, int]] = Field(
        ...,
        description="List of input edges as (node_name, index) tuples",
    )


StageModelType = TypeVar("StageModelType", bound=StageModel)

stage_models_list: TypeAlias = list[_stage_Models]  # pyright: ignore


#
class Graph(BaseModel):
    """Graph object to hold the pipeline information."""

    name: str = Field(..., description="Name of the graph")
    fs: int = Field(..., description="Sampling frequency for the graph")
    nodes: stage_models_list  # pyright: ignore
    inputs: list[Input]
    outputs: list[Output]


def path_encoder(obj):
    """Encode Path objects as strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DspJson(BaseModel):
    """Pydantic model of the JSON file describing a DSP graph."""

    ir_version: int
    producer_name: str
    producer_version: str
    graph: Graph
    # checksum: List

    def model_dump_xdsp(self, indent=2):
        """Dump model in xdsp format with specified indentation."""
        d = self.model_dump()
        # Move 'op_type' to the front
        for i, node in enumerate(d["graph"]["nodes"]):
            if "op_type" in node:
                items = [("op_type", node.pop("op_type"))] + list(node.items())
                d["graph"]["nodes"][i] = dict(items)
                # return json.dumps(dict(items))
        dump = json.dumps(d, indent=2, default=path_encoder)

        def compact_array_newlines(s):
            # Finds square brackets with their content (non-greedy, so not nested)
            pattern = re.compile(r"\[(.*?)\]", re.DOTALL)

            def repl(match):
                inner = match.group(1)
                if "[" in inner:
                    # catch the first node
                    return f"[{compact_array_newlines(inner + ']')}"
                elif "{" in inner or "}" in inner:
                    # catch other lists
                    return match.group(0)
                # Remove leading/trailing whitespace just inside the brackets
                stripped_inner = inner.strip()
                # Replace inner newlines+whitespace with a space
                compact_inner = re.sub(r"\s*\n\s*", " ", stripped_inner)
                return f"[{compact_inner}]"

            return pattern.sub(repl, s)

        return compact_array_newlines(dump)


def insert_forks(graph: Graph) -> Graph:
    """Automatically insert forks in the graph where edges have been used
    multiple times.
    """
    # Create a deep copy of the graph to avoid mutating the original
    new_graph = graph.model_copy(deep=True)
    # Map from edge name to a list of consumers (node or graph output, index, position)
    consumer_map: dict[tuple[str, int], list[tuple[str, Any, int]]] = defaultdict(list)
    # Build the consumer map for all node inputs
    for node_index, node in enumerate(new_graph.nodes):
        for pos, edge in enumerate(node.placement.input):
            consumer_map[tuple(edge)].append(("node", node_index, pos))
    # Add graph outputs as consumers
    for out_idx, out in enumerate(new_graph.outputs):
        for pos, edge in enumerate(out.input):
            consumer_map[tuple(edge)].append(("graph_output", out_idx, pos))
    all_edges: list[tuple[str, int]] = []
    for inp in new_graph.inputs:
        all_edges.extend([tuple(e) for e in inp.output])
    for out in new_graph.outputs:
        all_edges.extend([tuple(e) for e in out.input])
    for node in new_graph.nodes:
        all_edges.extend([tuple(e) for e in node.placement.input])
        # all_edges.extend([tuple(e) for e in node.placement.output])
    # Edge names are now strings, so no max_edge/new_edge_id logic
    producer_of_edge: dict[tuple[str, int], Optional[int]] = {}
    for inp in new_graph.inputs:
        for idx, edge in enumerate(inp.output):
            producer_of_edge[tuple(edge)] = None
    for idx, node in enumerate(new_graph.nodes):
        for e in node.placement.output:
            producer_of_edge[tuple(e)] = idx
    # Map from producer index to a list of Fork nodes to insert after it
    fork_nodes_by_producer: dict[Optional[int], list] = defaultdict(list)
    # For each edge with multiple consumers, insert a Fork node
    for edge, consumers in consumer_map.items():
        if len(consumers) > 1:
            num_consumers = len(consumers)
            new_edges = []
            for i in range(num_consumers):
                # Use the fork node name as the namespace
                new_edges.append(make_edge_name(f"AutoFork_{edge[0]}", i))
            for new_edge, (cons_type, consumer_idx, pos) in zip(new_edges, consumers):
                if cons_type == "node":
                    new_graph.nodes[consumer_idx].placement.input[pos] = new_edge
                elif cons_type == "graph_output":
                    new_graph.outputs[consumer_idx].input[pos] = new_edge
            # Create the Fork node and associate it with the producer
            fork_node_data = {
                "op_type": "Fork",
                "config": {"count": num_consumers},
                "placement": {
                    "input": [list(edge)],
                    "output": [list(e) for e in new_edges],
                    "name": f"AutoFork_{edge[0]}",
                    "thread": 0,
                },
            }
            fork_node = Fork(**fork_node_data)
            producer_idx = producer_of_edge.get(edge, None)
            fork_nodes_by_producer[producer_idx].append(fork_node)
    # Build the new node list, inserting Forks after their producers
    new_nodes = []
    if None in fork_nodes_by_producer:
        new_nodes.extend(fork_nodes_by_producer[None])
        del fork_nodes_by_producer[None]
    for idx, node in enumerate(new_graph.nodes):
        new_nodes.append(node)
        if idx in fork_nodes_by_producer:
            new_nodes.extend(fork_nodes_by_producer[idx])
    new_graph.nodes = new_nodes
    return new_graph


def stage_handle(model):
    """Get the function handle of a DSP Stage from its pydantic model."""
    return getattr(Stages, model.op_type)


def make_edge_name(node_name: str, idx: int) -> tuple[str, int]:
    """Return the canonical edge name as a tuple, e.g., ('NodeName', 0)."""
    return (node_name, idx)


def make_pipeline(json_obj: DspJson) -> Pipeline:
    """Create a Python DSP pipeline from a Pydantic model of the JSON file
    describing a DSP graph.
    """
    # Insert Fork nodes where needed to handle shared edges
    # graph = insert_forks(json_obj.graph)
    graph = json_obj.graph
    edge_map = {}  # edge_name (tuple) -> pipeline object

    # Collect all input edge names and total channels
    flat_input_edges = []
    total_channels = 0
    for inp in graph.inputs:
        flat_input_edges.extend([tuple(e) for e in inp.output])
        total_channels += len(inp.output)

    p, in_edges = Pipeline.begin(total_channels, fs=graph.fs)
    thread_max = max(node.placement.thread for node in graph.nodes) if graph.nodes else 0
    for _ in range(thread_max):
        p._add_thread()

    # Assign input edges
    for i, edge_name in enumerate(flat_input_edges):
        edge_map[edge_name] = in_edges[i]

    waiting_nodes = list(range(len(graph.nodes)))
    while waiting_nodes:
        node_idx = waiting_nodes[0]
        node = graph.nodes[node_idx]
        # Gather StageOutput objects for this node's inputs
        stage_inputs = []
        for edge_name in node.placement.input:
            stage_inputs.append(edge_map.get(tuple(edge_name)))
        if None in stage_inputs:
            waiting_nodes.pop(0)
            waiting_nodes.append(node_idx)
            continue
        # Combine all inputs into a StageOutputList
        stage_inputs = sum(stage_inputs, start=StageOutputList())
        # Get config dict for this node
        config = node.config if hasattr(node, "config") else {}
        if isinstance(config, BaseModel):
            config = config.model_dump()
        # Instantiate the stage and get its outputs
        node_output = p.stage(
            stage_handle(node),
            stage_inputs,
            node.placement.name,
            thread=node.placement.thread,
            **config,
        )
        # Set parameters if present
        if hasattr(node, "parameters"):
            p.stages[-1].set_parameters(node.parameters)
        # Map the outputs to the correct edge indices
        if len(node_output) != 0:
            for i in range(len(node.placement.output)):
                edge_name = tuple(node.placement.output[i])
                if edge_map.get(edge_name) is not None:
                    raise ValueError("Output already exists")
                edge_map[edge_name] = node_output[i]
        # Remove this node from waiting list
        waiting_nodes.pop(0)
    # Gather all output nodes for the pipeline
    output_nodes = []
    for edge_name in [tuple(e) for out in graph.outputs for e in out.input]:
        output_nodes.append(edge_map[edge_name])
    output_nodes = sum(output_nodes, start=StageOutputList())
    # Set the pipeline outputs
    p.set_outputs(output_nodes)
    return p


def update_pipeline(p: Pipeline, params: DspJson):
    """Update the pipeline with new DSP JSON parameters."""
    for stage in p.stages[1:]:
        if stage.name in ["pipeline", "dsp_thread"]:
            continue
        updated = False
        for node in params.graph.nodes:
            if node.placement.name == stage.label:
                updated = True
                if hasattr(node, "parameters"):
                    stage.set_parameters(node.parameters)
                break

        if not updated and "AutoFork" not in stage.label:
            warnings.warn(f"Stage {stage.label} could not be found in the JSON file")


def pipeline_to_dspjson(pipeline) -> DspJson:
    """Convert a Pipeline object to a DspJson object."""
    # Example: Extract graph-level info
    graph_name = getattr(pipeline, "_id")
    fs = getattr(pipeline, "fs")

    # Extract inputs and outputs
    inputs = [
        Input(
            name="inputs",
            channels=len(pipeline.i.edges),
        )
    ]
    outputs = [
        Output(name="outputs", input=[[f"{x.source.label}", x.source_index] for x in pipeline.o.edges])]

    # Extract nodes
    nodes = []
    for thread in pipeline.threads:
        for stage in thread._stages:
            op_type = type(stage).__name__
            if op_type in ["PipelineStage", "DSPThreadStage"]:
                continue

            stage_in = []
            for x in stage.i.edges:
                if x.source is not None:
                    stage_in.append([f"{x.source.label}", x.source_index])
                else:
                    stage_in.append([f"input", x.source_index])

            placement = {
                "name": stage.label or stage.name,
                "input": stage_in,
                "thread": thread.id,
            }

            node_dict = {
                "op_type": op_type,
                "placement": placement,  # Should be a dict or Pydantic model
            }
            if hasattr(stage, "config"):
                node_dict["config"] = stage.config
            if hasattr(stage, "parameters"):
                node_dict["parameters"] = stage.parameters
            # Convert to the correct Pydantic model for the node
            node_model_cls = type(stage.model) if hasattr(stage, "model") else None
            if node_model_cls:
                node = node_model_cls(**node_dict)
            else:
                node = node_dict  # fallback, but ideally use the model
            nodes.append(node)

    graph = Graph(
        name=graph_name,
        fs=fs,
        nodes=nodes,
        inputs=inputs,
        outputs=outputs,
    )

    # Fill in DspJson fields
    dsp_json = DspJson(
        ir_version=1,
        producer_name="pipeline_to_dspjson",
        producer_version="1.0",
        graph=graph,
    )
    return dsp_json
