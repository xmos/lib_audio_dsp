"""Shut up ruff."""

import copy
import io
import json
import logging
import tempfile
import traceback
import wave
from pathlib import Path
from typing import Annotated, Optional, Type, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    Field,
)

import audio_dsp.stages as Stages
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.design.stage import StageOutputList
from audio_dsp.models.stage import StageModel, all_models, edgeProducerBaseModel

BAD_NAMES = ["CascadedBiquads"]

_stage_Models = Annotated[
    Union[tuple(i for i in all_models().values() if i.__name__ not in BAD_NAMES)],
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
        {"config": {"count": 1}, "op_type": "Fork", "placement": {"input": [3], "name": "Fork", "output": [4, 5], "thread": 0}},
      ]
      "output": {"channels": 2, "input": [4, 5], "name": "stereo_out"}
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


def stage_handle(model):
    """Shut up ruff."""
    return getattr(Stages, model.op_type)


def make_pipeline(json_obj: DspJson) -> Pipeline:
    """Shut up ruff."""
    graph = json_obj.graph

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
