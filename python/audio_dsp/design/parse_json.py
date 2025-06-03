"""Functions to convert JSON files to Python DSP pipelines."""

from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Annotated, Any, Optional, Union, List

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.stages as Stages
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.design.stage import StageOutputList, all_stages
from audio_dsp.models.signal_chain import Fork
from audio_dsp.models.stage import all_models
import json
import re
import warnings

BAD_NAMES = ["CascadedBiquads"]

_stage_Models = Annotated[
    Union[
        tuple((SkipJsonSchema[i] if i.__name__ in BAD_NAMES else i) for i in all_models().values())
    ],
    Field(discriminator="op_type"),
]


class Input(BaseModel, extra="ignore"):
    """Pydantic model of the inputs to a DSP graph."""

    name: str = Field(..., description="Name of the input")
    output: list[int] = Field(
        ...,
        description="List of output edges (1 edge for mono, 2 for stereo)",
        min_length=1,
        max_length=2,
    )


class Output(BaseModel, extra="ignore"):
    """Pydantic model of the outputs of a DSP graph."""

    name: str = Field(..., description="Name of the output")
    input: list[int] = Field(
        ...,
        description="List of input edges (1 edge for mono, 2 for stereo)",
        min_length=1,
        max_length=2,
    )


class Graph(BaseModel):
    """
    Graph object to hold the pipeline.

    Examples
    --------
    1. EQ + Reverb Example:

    ```json
    {
      "name": "EQ + Reverb Example",
      "fs": 48000,
      "nodes": [
        {"placement": {"input": [0, 1], "output": [2, 3], "name": "VolumeIn", "thread": 0}, "op_type": "VolumeControl"},
        {"op_type": "ParametricEq", "placement": {"input": [2, 3], "output": [4, 5], "name": "PEQ", "thread": 0}},
        {"op_type": "ReverbPlateStereo", "config": {"predelay": 30}, "placement": {"input": [4, 5], "output": [6, 7], "name": "StereoReverb", "thread": 0}}
      ],
      "inputs": [
        {"name": "audio_in", "output": [0, 1]}
      ],
      "outputs": [
        {"name": "audio_out", "input": [6, 7]}
      ]
    }
    ```

    2. Stereo Mixer with Volume:

    ```json
    {
      "name": "Stereo Mixer with Volume",
      "fs": 48000,
      "inputs": [
        {"name": "stereo_in", "output": [0, 1]}
      ],
      "nodes": [
        {"op_type": "Mixer", "placement": {"input": [0, 1], "name": "Mixer", "output": [2], "thread": 0}},
        {"op_type": "VolumeControl", "placement": {"input": [2], "name": "Volume", "output": [3], "thread": 0}}
      ],
      "outputs": [
        {"name": "stereo_out", "input": [3, 3]}
      ]
    }
    ```
    """

    name: str = Field(..., description="Name of the graph")
    fs: int = Field(..., description="Sampling frequency for the graph")
    nodes: list[_stage_Models]  # pyright: ignore
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

    def model_dump_xdsp(self, indent):
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
    new_graph = graph.model_copy(deep=True)
    consumer_map: dict[int, list[tuple[str, Any, int]]] = defaultdict(list)
    for node_index, node in enumerate(new_graph.nodes):
        for pos, edge in enumerate(node.placement.input):
            consumer_map[edge].append(("node", node_index, pos))
    for out_idx, out in enumerate(new_graph.outputs):
        for pos, edge in enumerate(out.input):
            consumer_map[edge].append(("graph_output", out_idx, pos))
    all_edges: list[int] = []
    for inp in new_graph.inputs:
        all_edges.extend(inp.output)
    for out in new_graph.outputs:
        all_edges.extend(out.input)
    for node in new_graph.nodes:
        all_edges.extend(node.placement.input)
        all_edges.extend(node.placement.output)
    max_edge = max(all_edges) if all_edges else -1
    new_edge_id = max_edge + 1
    producer_of_edge: dict[int, Optional[int]] = {}
    for inp in new_graph.inputs:
        for edge in inp.output:
            producer_of_edge[edge] = None
    for idx, node in enumerate(new_graph.nodes):
        for e in node.placement.output:
            producer_of_edge[e] = idx
    fork_nodes_by_producer: dict[Optional[int], list] = defaultdict(list)
    for edge, consumers in consumer_map.items():
        if len(consumers) > 1:
            num_consumers = len(consumers)
            new_edges = []
            for _ in range(num_consumers):
                new_edges.append(new_edge_id)
                new_edge_id += 1
            for new_edge, (cons_type, consumer_idx, pos) in zip(new_edges, consumers):
                if cons_type == "node":
                    new_graph.nodes[consumer_idx].placement.input[pos] = new_edge
                elif cons_type == "graph_output":
                    new_graph.outputs[consumer_idx].input[pos] = new_edge
            fork_node_data = {
                "op_type": "Fork",
                "config": {"count": num_consumers},
                "placement": {
                    "input": [edge],
                    "output": new_edges,
                    "name": f"AutoFork_{edge}",
                    "thread": 0,
                },
            }
            fork_node = Fork(**fork_node_data)
            producer_idx = producer_of_edge.get(edge, None)
            fork_nodes_by_producer[producer_idx].append(fork_node)
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


def make_pipeline(json_obj: DspJson) -> Pipeline:
    """Create a Python DSP pipeline from a Pydantic model of the JSON file
    describing a DSP graph.
    """
    graph = insert_forks(json_obj.graph)
    flat_input_edges: list[int] = []
    total_channels = 0
    for inp in graph.inputs:
        flat_input_edges.extend(inp.output)
        total_channels += len(inp.output)
    flat_output_edges: list[int] = []
    for out in graph.outputs:
        flat_output_edges.extend(out.input)
    edgelist: list[int] = []
    edgelist.extend(flat_input_edges)
    edgelist.extend(flat_output_edges)
    for node in graph.nodes:
        edgelist.extend(node.placement.input)
        edgelist.extend(node.placement.output)
    max_edge = max(edgelist) if edgelist else -1
    p, in_edges = Pipeline.begin(total_channels, fs=graph.fs)
    thread_max = max(node.placement.thread for node in graph.nodes) if graph.nodes else 0
    for _ in range(thread_max):
        p._add_thread()
    edge_list = [None] * (max_edge + 1)
    for i, edge in enumerate(flat_input_edges):
        edge_list[edge] = in_edges[i]
    waiting_nodes = list(range(len(graph.nodes)))
    while waiting_nodes:
        node_idx = waiting_nodes[0]
        node = graph.nodes[node_idx]
        stage_inputs = []
        for i in node.placement.input:
            stage_inputs.append(edge_list[i])
        if None in stage_inputs:
            waiting_nodes.pop(0)
            waiting_nodes.append(node_idx)
            continue
        stage_inputs = sum(stage_inputs, start=StageOutputList())
        config = node.config if hasattr(node, "config") else {}
        if isinstance(config, BaseModel):
            config = config.model_dump()
        node_output = p.stage(
            stage_handle(node),
            stage_inputs,
            node.placement.name,
            thread=node.placement.thread,
            **config,
        )
        if hasattr(node, "parameters"):
            p.stages[-1].set_parameters(node.parameters)
        if len(node_output) != 0:
            for i in range(len(node.placement.output)):
                if edge_list[node.placement.output[i]] is not None:
                    raise ValueError("Output already exists")
                edge_list[node.placement.output[i]] = node_output[i]
        waiting_nodes.pop(0)
    output_nodes = []
    for edge in flat_output_edges:
        output_nodes.append(edge_list[edge])
    output_nodes = sum(output_nodes, start=StageOutputList())
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
    """
    Convert a Pipeline object to a DspJson object.
    """
    # Example: Extract graph-level info
    graph_name = getattr(pipeline, "name", "Generated DSP Graph")
    fs = getattr(pipeline, "fs", 48000)

    # Extract inputs and outputs
    inputs = []
    for inp in getattr(pipeline, "inputs", []):
        inputs.append(Input(name=inp["name"], output=inp["output"]))

    outputs = []
    for out in getattr(pipeline, "outputs", []):
        outputs.append(Output(name=out["name"], input=out["input"]))

    # Extract nodes
    nodes = []
    for stage in pipeline.stages:  # skip pipeline root if needed
        op_type = type(stage).__name__
        if op_type in ["PipelineStage", "DSPThreadStage"]:
            continue
        node_dict = {
            "op_type": op_type,
            "placement": stage.placement,  # Should be a dict or Pydantic model
        }
        if hasattr(stage, "config") and isinstance(stage.config, BaseModel):
            node_dict["config"] = stage.config
        if hasattr(stage, "parameters") and isinstance(stage.parameters, BaseModel):
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


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="JSON-to-DSP pipeline generator")
    # parser.add_argument(
    #     "json_path", type=Path, help="path to the JSON describing the DSP pipeline"
    # )
    # parser.add_argument("out_path", type=Path, help="path for the generated DSP code output")
    # args = parser.parse_args()

    # output_path = Path(args.out_path)
    # json_path = Path(args.json_path)

    # json_path = Path(r"C:\Users\allanskellett\Documents\051_dsp_txt\dsp_lang_1.json")
    json_path = Path(r"C:\Users\allanskellett\Documents\040_dsp_ultra\scio_0_new_forks.json")
    output_path = "tmpdir"
    json_obj = DspJson.model_validate_json(json_path.read_text())
    p = make_pipeline(json_obj)
    generate_dsp_main(p, output_path)
    p.draw(Path(output_path, "dsp_pipeline"))
