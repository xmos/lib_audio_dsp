from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.stages as Stages
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.design.stage import StageOutputList
from audio_dsp.models.signal_chain import Fork
from audio_dsp.models.stage import StageModel, all_models

BAD_NAMES = ["CascadedBiquads", "Fork"]

_stage_Models = Annotated[
    Union[
        tuple((SkipJsonSchema[i] if i.__name__ in BAD_NAMES else i) for i in all_models().values())
    ],
    Field(discriminator="op_type"),
]


class Input(BaseModel, extra="ignore"):
    name: str = Field(..., description="Name of the input")
    output: list[int] = Field(
        ...,
        description="List of output edges (1 edge for mono, 2 for stereo)",
        min_length=1,
        max_length=2,
    )


class Output(BaseModel, extra="ignore"):
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

    Examples:
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
    nodes: list[_stage_Models]
    inputs: list[Input]
    outputs: list[Output]


class DspJson(BaseModel):
    ir_version: int
    producer_name: str
    producer_version: str
    graph: Graph


def insert_forks(graph: Graph) -> Graph:
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
    return getattr(Stages, model.op_type)


def make_pipeline(json_obj: DspJson) -> Pipeline:
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


def _test_no_shared_edge():
    json_str = """
    {
      "name": "No shared edge",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "VolumeControl_1",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [1]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_no_shared_edge"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_shared_edge_from_graph_input():
    json_str = """
    {
      "name": "Shared edge from graph input",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "VolumeControl_A",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [0],
            "output": [2],
            "name": "Mixer_B",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [1]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_shared_edge_from_graph_input"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_shared_edge_from_producer_node():
    json_str = """
    {
      "name": "Shared edge from producer node",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "Producer",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1],
            "output": [2],
            "name": "Consumer_1",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1],
            "output": [3],
            "name": "Consumer_2",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [2]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_shared_edge_from_producer_node"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_shared_edge_with_graph_output():
    json_str = """
    {
      "name": "Shared edge with graph output",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "Producer",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1],
            "output": [2],
            "name": "Consumer",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [1]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_shared_edge_with_graph_output"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_again():
    json_str = """
    {
      "name": "Stereo Mixer with Volume",
      "fs": 48000,
      "inputs": [{
          "name": "stereo_in",
          "output": [0, 1]
      }],
      "nodes": [
        {"op_type": "Mixer", "placement": {"input": [0, 1], "name": "Mixer", "output": [2], "thread": 0}},
        {"op_type": "VolumeControl", "placement": {"input": [2], "name": "Volume", "output": [3], "thread": 0}}
      ],
      "outputs": [{
          "name": "stereo_out",
          "input": [3, 3]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_again"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_multiple_inputs_outputs_non_shared():
    """
    Test a graph with two inputs (one mono and one stereo) and two outputs (one mono and one stereo)
    where no edge is shared between consumers.
    """
    json_str = """
    {
      "name": "Multiple Inputs and Outputs Non-Shared Test",
      "fs": 48000,
      "inputs": [
        {"name": "mono_in", "output": [0]},
        {"name": "stereo_in", "output": [1, 2]}
      ],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [3],
            "name": "VolumeControl_A",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1, 2],
            "output": [4],
            "name": "Mixer_B",
            "thread": 0
          }
        }
      ],
      "outputs": [
        {"name": "mono_out", "input": [3]},
        {"name": "stereo_out", "input": [4, 3]}
      ]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_multiple_inputs_outputs_non_shared"))
    new_graph = insert_forks(graph)
    print("Test: Multiple Inputs and Outputs Non-Shared Test")
    print(f"Before insert_forks: {graph.model_dump_json(indent=2)}")
    print(f"After insert_forks: {new_graph.model_dump_json(indent=2)}")


def _test_multiple_inputs_outputs_shared():
    """
    Test a graph with two inputs and two outputs where a node produces an output
    that is shared by both graph outputs. A Fork should be inserted.
    """
    json_str = """
    {
      "name": "Multiple Inputs and Outputs Shared Test",
      "fs": 48000,
      "inputs": [
        {"name": "input1", "output": [0]},
        {"name": "input2", "output": [1]}
      ],
      "nodes": [
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [0, 1],
            "output": [2],
            "name": "Mixer",
            "thread": 0
          }
        }
      ],
      "outputs": [
        {"name": "output1", "input": [2]},
        {"name": "output2", "input": [2]}
      ]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_multiple_inputs_outputs_shared"))
    new_graph = insert_forks(graph)
    print("Test: Multiple Inputs and Outputs Shared Test")
    print(f"Before insert_forks: {graph.model_dump_json(indent=2)}")
    print(f"After insert_forks: {new_graph.model_dump_json(indent=2)}")


if __name__ == "__main__":
    # Uncomment the tests you wish to run:
    _test_again()
    _test_shared_edge_from_producer_node()
    _test_shared_edge_with_graph_output()
    _test_shared_edge_from_graph_input()
    _test_no_shared_edge()
    _test_multiple_inputs_outputs_non_shared()
    _test_multiple_inputs_outputs_shared()
    # pprint(Graph.model_json_schema())
    print("All tests passed.")
