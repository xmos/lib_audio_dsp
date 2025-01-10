from pydantic import BaseModel, RootModel, Field, create_model, TypeAdapter, field_validator, PrivateAttr, computed_field, ConfigDict
from typing import Literal, Annotated, List, Union, Optional
from annotated_types import Len
from functools import partial
from pathlib import Path
import itertools
import numpy as np

from audio_dsp.design.stage import Stage, all_stages
import audio_dsp.stages as Stages

_stages_list = tuple(all_stages().keys())

class nodeBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input: Optional[list[int]] = None
    output: Optional[list[int]] = None

    @field_validator("input", "output", mode="before")
    def _single_to_list(cls, value: Union[int, list]) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]

class Node(nodeBaseModel):
    input: list[int]
    output: list[int]
    name: str
    op_type: Literal[_stages_list]
    thread: int
    parameters: Optional[dict[str, float]] = None
    
    @computed_field
    def _stage_handle(self) -> Stage:
        return getattr(Stages, self.op_type)




class Input(nodeBaseModel):
    name: str
    output: list[int]
    channels: int
    fs: int

class Output(nodeBaseModel):
    name: str
    input: list[int]
    channels: int
    fs: Optional[int] = None

class Graph(BaseModel):
    name: str
    nodes: list[Node]
    input: Input
    output: Output

class DspJson(BaseModel):
    ir_version: int
    producer_name: str
    producer_version: str
    graph: Graph


if __name__ == "__main__":
    # json_obj = DspJson.model_validate_json(Path(r"C:\Users\allanskellett\Documents\051_dsp_txt\dsp_lang_1.json").read_text())
    json_obj = DspJson.model_validate_json(Path(r"C:\Users\allanskellett\Documents\040_dsp_ultra\frj_0.json").read_text())

    graph = json_obj.graph

    # get flat list of nodes
    edgelist = graph.input.output + graph.output.input
    for i in graph.nodes:
        [edgelist.append(n) for n in i.input]
        [edgelist.append(n) for n in i.output]

    uniq_edges, edge_counts = np.unique(edgelist, return_counts=True)
    # assert np.all(edge_counts == 2), "All nodes should be used twice in the graph"
    edge_list = [None]*len(uniq_edges)

    from audio_dsp.design.pipeline import Pipeline, generate_dsp_main

    p, in_edges = Pipeline.begin(graph.input.channels, fs=graph.input.fs)

    for n in range(graph.input.channels):
        edge_list[n] = in_edges[n]

    node_outputs = [None]*len(graph.nodes)

    waiting_nodes = list(range(len(graph.nodes)))

    while waiting_nodes:
        this_node = graph.nodes[waiting_nodes[0]]
        stage_inputs = []
        for i in this_node.input:
            stage_inputs.append(edge_list[i])

        if None in stage_inputs:
            this_node = waiting_nodes.pop(0)
            waiting_nodes.append(this_node)
            continue

        stage_inputs = sum(stage_inputs)
        if this_node.op_type == "Fork":
            count = len(this_node.output)//len(this_node.input)
            node_output = p.stage(this_node._stage_handle, stage_inputs, this_node.name, count=count)
        else:
            node_output = p.stage(this_node._stage_handle, stage_inputs, this_node.name)

        if len(node_output) != 0:
            for i in range(len(this_node.output)):
                if edge_list[this_node.output[i]] is None:
                    edge_list[this_node.output[i]] = node_output[i]
                else:
                    assert False , "oops"
    
        waiting_nodes.pop(0)

    output_nodes = []
    for i in graph.output.input:
        output_nodes.append(edge_list[i])
    output_nodes = sum(output_nodes)

    p.set_outputs(output_nodes)

    pass