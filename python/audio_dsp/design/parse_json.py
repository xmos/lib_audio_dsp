from pydantic import BaseModel, RootModel, Field, create_model, TypeAdapter, field_validator, PrivateAttr, computed_field, ConfigDict
from typing import Literal, Annotated, List, Union, Optional, Any
from annotated_types import Len
from functools import partial
from pathlib import Path
import itertools
import numpy as np

from audio_dsp.design.stage import Stage, all_stages, edgeProducerBaseModel
import audio_dsp.stages as Stages

_stage_dict = all_stages()
_stages_list = tuple(all_stages().keys())
_stages_2 = Union[tuple(all_stages().values())]
_stage_Models = Annotated[Union[tuple(i.Model for i in all_stages().values())], Field(discriminator="op_type")]

class Input(edgeProducerBaseModel):
    name: str
    output: list[int]
    channels: int
    fs: int

class Output(edgeProducerBaseModel):
    name: str
    input: list[int]
    channels: int
    fs: Optional[int] = None

class Graph(BaseModel):
    name: str
    nodes: List[_stage_Models]
    input: Input
    output: Output

class DspJson(BaseModel):
    ir_version: int
    producer_name: str
    producer_version: str
    graph: Graph

def stage_handle(model):
    return getattr(Stages, model.op_type)

if __name__ == "__main__":
    from audio_dsp.design.pipeline import Pipeline, generate_dsp_main

    # json_obj = DspJson.model_validate_json(Path(r"C:\Users\allanskellett\Documents\051_dsp_txt\dsp_lang_1.json").read_text())
    json_obj = DspJson.model_validate_json(Path(r"C:\Users\allanskellett\Documents\040_dsp_ultra\frj_0.json").read_text())

    graph = json_obj.graph

    # get flat list of edges and threads
    edgelist = graph.input.output + graph.output.input
    threadlist = []
    for i in graph.nodes:
        [edgelist.append(n) for n in i.input]
        [edgelist.append(n) for n in i.output]
        threadlist.append(i.thread)

    p, in_edges = Pipeline.begin(graph.input.channels, fs=graph.input.fs)

    # make the right number of threads
    for n in range(max(threadlist)):
        p._add_thread()
    
    # make edge list, populate first N with inputs
    edge_list = [None]*(max(edgelist) + 1)
    for n in range(graph.input.channels):
        edge_list[n] = in_edges[n]

    waiting_nodes = list(range(len(graph.nodes)))

    while waiting_nodes:
        this_node = graph.nodes[waiting_nodes[0]]

        # get node inputs
        stage_inputs = []
        for i in this_node.input:
            stage_inputs.append(edge_list[i])

        if None in stage_inputs:
            # input doesn't exist yet, try next node, add this node to the end
            this_node = waiting_nodes.pop(0)
            waiting_nodes.append(this_node)
            continue

        stage_inputs = sum(stage_inputs)
        node_output = p.stage(stage_handle(this_node), stage_inputs, this_node.name,
                              thread=this_node.thread, **this_node.config.dict())

        p.stages[-1].set_parameters(this_node.parameters)

        # if has outputs, add to edge to edge list- nothing should be there!
        if len(node_output) != 0:
            for i in range(len(this_node.output)):
                if edge_list[this_node.output[i]] is None:
                    edge_list[this_node.output[i]] = node_output[i]
                else:
                    assert False , "oops"
    
        # done so pop
        waiting_nodes.pop(0)


    # setup the output
    output_nodes = [None] * graph.output.channels
    for i in range(len(graph.output.input)):
        output_nodes[i] = edge_list[graph.output.input[i]]
    output_nodes = sum(output_nodes)
    p.set_outputs(output_nodes)

    pass