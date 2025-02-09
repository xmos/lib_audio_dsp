"""Shut up ruff."""

from collections import defaultdict
from typing import Annotated, Any, Optional, Union

from pydantic import (
    BaseModel,
    Field,
)
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.stages as Stages
from audio_dsp.models.signal_chain import Fork
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.design.stage import StageOutputList
from audio_dsp.models.stage import StageModel, all_models, edgeProducerBaseModel

BAD_NAMES = ["CascadedBiquads", "Fork"]

_stage_Models = Annotated[
    Union[
        tuple((SkipJsonSchema[i] if i.__name__ in BAD_NAMES else i) for i in all_models().values())
    ],
    Field(discriminator="op_type"),
]


class Input(edgeProducerBaseModel, extra="ignore"):
    name: str = Field(..., description="Name of the input")
    output: list[int] = Field(
        default_factory=list,
        description="List of output edges, should be a range of the number of channels",
    )
    channels: int = Field(..., description="Number of input channels")
    fs: int = Field(..., description="Sampling frequency in Hz")


class Output(edgeProducerBaseModel, extra="ignore"):
    name: str = Field(..., description="Name of the output")
    input: list[int] = Field(
        default_factory=list,
        description="List of input edges, should be a range of the number of channels. Input edges to this stage can not be used elsewhere, same as with any other stage.",
    )
    channels: int = Field(..., description="Number of output channels")
    fs: Optional[int] = None


class Graph(BaseModel):
    """
    Graph object to hold the pipeline.

    Pay attention to the field definitions of the nodes, including the number of input and output edges for each node (and edge is a channel).

    Examples:
    --------

    1. EQ + Reverb Example:

    ```json
    {
      "name": "EQ + Reverb Example",
      "nodes": [
        {"placement": {"input": [0, 1], "output": [2, 3], "name": "VolumeIn", "thread": 0}, "op_type": "VolumeControl"},
        {"op_type": "ParametricEq", "placement": {"input": [2, 3], "output": [4, 5], "name": "PEQ", "thread": 0}},
        {"op_type": "ReverbPlateStereo", "config": {"predelay": 30}, "placement": {"input": [4, 5], "output": [6, 7], "name": "StereoReverb", "thread": 0}},
      ],
      "input": {"name": "audio_in", "output": [0, 1], "channels": 2, "fs": 48000},
      "output": {"name": "audio_out", "input": [6, 7], "channels": 2}
    }
    ```

    2. Stereo Mixer with Volume:

    ```json
    {
      "name": "Stereo Mixer with Volume",
      "input": {"channels": 2, "fs": 48000, "name": "stereo_in", "output": [0, 1]},
      "nodes": [
        {"op_type": "Mixer", "placement": {"input": [0, 1], "name": "Mixer", "output": [2], "thread": 0}},
        {"op_type": "VolumeControl", "placement": {"input": [2], "name": "Volume", "output": [3], "thread": 0}},
      ]
      "output": {"channels": 2, "input": [3, 3], "name": "stereo_out"}
    }
    """

    name: str = Field(
        ...,
        description="Name of the graph, should describe what the graph does. Space are allowed.",
    )
    nodes: list[_stage_Models]  # type: ignore
    input: Input
    output: Output


class DspJson(BaseModel):
    ir_version: int
    producer_name: str
    producer_version: str
    graph: Graph


def insert_forks(graph: Graph) -> Graph:
    """
    Return a new Graph in which every stage input comes from a unique edge.
    That is, if an output edge is used by more than one consumer (a node’s input or the graph's output),
    a Fork node is inserted to split that edge into unique ones.

    For each edge E with >1 consumer:
      - Create new unique edge IDs (one per consumer).
      - Update each consumer’s placement.input (or graph.output.input) to use one of the new edges.
      - Insert a new Fork node with placement.input = [E] and placement.output = [new_edge1, new_edge2, ...].
      - Importantly, insert the Fork node immediately after the node that produced edge E.
        If E is produced by the graph input, insert the Fork node at the beginning.

    Args:
        graph (Graph): The original graph without manually inserted Fork nodes.

    Returns
    -------
        Graph: A new Graph with Fork nodes inserted so that no stage input re-uses an output edge.
    """
    new_graph = graph.model_copy(deep=True)
    consumer_map: dict[int, list[tuple[str, Any, int]]] = defaultdict(list)

    for node_index, node in enumerate(new_graph.nodes):
        for pos, edge in enumerate(node.placement.input):
            consumer_map[edge].append(("node", node_index, pos))

    for pos, edge in enumerate(new_graph.output.input):
        consumer_map[edge].append(("graph_output", None, pos))

    all_edges = list(new_graph.input.output) + list(new_graph.output.input)
    for node in new_graph.nodes:
        all_edges.extend(node.placement.input)
        all_edges.extend(node.placement.output)
    max_edge = max(all_edges) if all_edges else -1
    new_edge_id = max_edge + 1

    producer_of_edge: dict[int, Optional[int]] = {}
    for edge in new_graph.input.output:
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
                    new_graph.output.input[pos] = new_edge

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
    """Shut up ruff."""
    return getattr(Stages, model.op_type)


def make_pipeline(json_obj: DspJson) -> Pipeline:
    """Shut up ruff."""
    graph = insert_forks(json_obj.graph)

    # get flat list of edges and threads
    edgelist = graph.input.output + graph.output.input
    threadlist = []
    for i in graph.nodes:
        [edgelist.append(n) for n in i.placement.input]
        [edgelist.append(n) for n in i.placement.output]
        threadlist.append(i.placement.thread)

    p, in_edges = Pipeline.begin(graph.input.channels, fs=graph.input.fs)

    # make the right number of threads
    for n in range(max(threadlist)):
        p._add_thread()

    # make edge list, populate first N with inputs
    edge_list = [None] * (max(edgelist) + 1)
    for n in range(graph.input.channels):
        edge_list[n] = in_edges[n]

    waiting_nodes = list(range(len(graph.nodes)))

    while waiting_nodes:
        this_node = graph.nodes[waiting_nodes[0]]

        # get node inputs
        stage_inputs = []
        for i in this_node.placement.input:
            stage_inputs.append(edge_list[i])

        if None in stage_inputs:
            # input doesn't exist yet, try next node, add this node to the end
            this_node = waiting_nodes.pop(0)
            waiting_nodes.append(this_node)
            continue

        stage_inputs = sum(stage_inputs, start=StageOutputList())
        node_output = p.stage(
            stage_handle(this_node),
            stage_inputs,
            this_node.placement.name,
            thread=this_node.placement.thread,
            **dict(this_node.config if hasattr(this_node, "config") else {}),
        )

        if hasattr(this_node, "parameters"):
            p.stages[-1].set_parameters(this_node.parameters)

        # if has outputs, add to edge to edge list- nothing should be there!
        if len(node_output) != 0:
            for i in range(len(this_node.placement.output)):
                if edge_list[this_node.placement.output[i]] is not None:
                    raise ValueError("Output already exists")
                edge_list[this_node.placement.output[i]] = node_output[i]

        # done so pop
        waiting_nodes.pop(0)

    # setup the output
    output_nodes = [None] * graph.output.channels
    for i in range(len(graph.output.input)):
        output_nodes[i] = edge_list[graph.output.input[i]]
    output_nodes = sum(output_nodes, start=StageOutputList())
    p.set_outputs(output_nodes)

    return p


def _test_no_shared_edge():
    json_str = """
    {
      "name": "No shared edge",
      "input": {
        "name": "audio_in",
        "output": [0],
        "channels": 1,
        "fs": 44100
      },
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
      "output": {
        "name": "audio_out",
        "input": [1],
        "channels": 1
      }
    }
    """
    graph = Graph.model_validate_json(json_str)
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_shared_edge_from_graph_input():
    json_str = """
    {
      "name": "Shared edge from graph input",
      "input": {
        "name": "audio_in",
        "output": [0],
        "channels": 1,
        "fs": 44100
      },
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
      "output": {
        "name": "audio_out",
        "input": [1],
        "channels": 1
      }
    }
    """
    graph = Graph.model_validate_json(json_str)
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_shared_edge_from_producer_node():
    json_str = """
    {
      "name": "Shared edge from producer node",
      "input": {
        "name": "audio_in",
        "output": [0],
        "channels": 1,
        "fs": 44100
      },
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
      "output": {
        "name": "audio_out",
        "input": [2],
        "channels": 1
      }
    }
    """
    graph = Graph.model_validate_json(json_str)
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_shared_edge_with_graph_output():
    json_str = """
    {
      "name": "Shared edge with graph output",
      "input": {
        "name": "audio_in",
        "output": [0],
        "channels": 1,
        "fs": 44100
      },
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
      "output": {
        "name": "audio_out",
        "input": [1],
        "channels": 1
      }
    }
    """
    graph = Graph.model_validate_json(json_str)
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def _test_output_twice():
    json_str = """
    {
      "name": "Stereo Mixer with Volume",
      "input": {"channels": 2, "fs": 48000, "name": "stereo_in", "output": [0, 1]},
      "nodes": [
        {"op_type": "Mixer", "placement": {"input": [0, 1], "name": "Mixer", "output": [2], "thread": 0}},
        {"op_type": "VolumeControl", "placement": {"input": [2], "name": "Volume", "output": [3], "thread": 0}}
      ],
      "output": {"channels": 2, "input": [3, 3], "name": "stereo_out"}
    }
    """
    graph = Graph.model_validate_json(json_str)
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


# ------------------------------------------------------------
# Run Tests
# ------------------------------------------------------------
if __name__ == "__main__":
    _test_no_shared_edge()
    _test_shared_edge_from_graph_input()
    _test_shared_edge_from_producer_node()
    _test_shared_edge_with_graph_output()
    _test_output_twice()
    print("All tests passed.")
