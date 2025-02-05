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


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies and authentication headers
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

global_graph: Optional[Graph] = None
global_pipeline: Optional[Pipeline] = None


@app.get("/schema/graph")
def get_dsp_json_schema():
    """
    Return JSON schema for DspJson with the 'parameters' field
    stripped out of all models under _stage_Models.
    """
    schema = copy.deepcopy(Graph.model_json_schema())
    return JSONResponse(schema)


@app.get("/schema/params")
def get_params_schema():
    """Return JSON schema for the parameters of each stage."""
    params = {
        k: n.model_fields["parameters"].annotation.model_json_schema()
        for k, n in all_models().items()
        if (
            hasattr(n.model_fields, "parameters")
            and hasattr(n.model_fields["parameters"].annotation, "model_json_schema")
        )
    }
    return JSONResponse(params)


@app.post("/graph/params")
def set_parameters(params: dict):
    """Set the parameters of the globally stored graph."""
    global global_graph
    if global_graph is None:
        raise HTTPException(status_code=400, detail="No graph has been set.")
    for node in global_graph.nodes:
        if node.placement.name in params and hasattr(node, "parameters"):
            node.parameters = node.parameters.__class__(**params[node.placement.name])
    global global_pipeline
    global_pipeline = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="0.1", graph=global_graph)
    )
    return {"message": "Parameters successfully set."}


@app.get("/graph/params")
def get_parameters():
    """Get the parameters of the globally stored graph."""
    global global_graph
    if global_graph is None:
        raise HTTPException(status_code=400, detail="No graph has been set.")
    return {node.placement.name: node.parameters for node in global_graph.nodes}


@app.post("/graph/audio")
async def run_audio(file: UploadFile = File(...)):
    """Run the audio through the pipeline."""
    global global_pipeline
    if global_pipeline is None:
        raise HTTPException(status_code=400, detail="No graph has been set.")
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    try:
        contents = await file.read()

        # Read the WAV file properties
        with wave.open(io.BytesIO(contents), "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            contents = wav_file.readframes(wav_file.getnframes())

        # Convert bytes to numpy array
        if sample_width == 2:  # 16-bit audio
            audio_data = np.frombuffer(contents, dtype=np.int16)
        elif sample_width == 4:  # 32-bit audio
            audio_data = np.frombuffer(contents, dtype=np.int32)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported sample width: {sample_width}"
            )

        # Reshape the audio data
        audio_data = audio_data.reshape(-1, n_channels)

        # If the pipeline expects 2 channels but we have 1, duplicate the channel
        assert global_graph is not None
        if len(global_graph.input.output) == 2 and n_channels == 1:
            audio_data = np.column_stack((audio_data, audio_data))

        max_value = 2 ** (8 * sample_width - 1)
        audio_data = audio_data.astype(np.float32) / max_value  # Process the audio
        executor = global_pipeline.executor().process
        a = executor(audio_data)
        data = a.data
        fs = a.fs
        sim_out = data
        output_scaled = (sim_out * 32767.0).astype(np.int16)
        output_bytes = output_scaled.tobytes()

        # Create a new WAV file in memory
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, "wb") as output_wav:
            output_wav.setnchannels(sim_out.shape[1])
            output_wav.setsampwidth(2)  # 16-bit audio
            output_wav.setframerate(fs)
            output_wav.writeframes(output_bytes)

        output_buffer.seek(0)
        return Response(content=output_buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        logging.exception("Error processing audio")
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph/set")
def set_graph(graph: Graph):
    """Set the global graph after validating it."""
    global global_graph
    global global_pipeline
    try:
        json_data = DspJson(
            ir_version=1, producer_name="test", producer_version="0.1", graph=graph
        )
        global_pipeline = make_pipeline(json_data)  # Validate the graph
        global_graph = graph  # Store it globally after successful validation
        # create the graph stage schema to be used in set_parameters
        schema_dict = {
            node.placement.name: node.parameters.__class__.model_json_schema()
            for node in graph.nodes
            if hasattr(node, "parameters")
        }
        return JSONResponse(schema_dict)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "traceback": traceback.format_exc()}, status_code=400
        )


@app.get("/graph/get")
def get_graph():
    """Retrieve the currently stored graph."""
    if global_graph is None:
        raise HTTPException(status_code=404, detail="No graph is set.")
    return global_graph


@app.post("/graph/render")
def render_dsp():
    """Render the globally stored DSP pipeline diagram as an SVG image."""
    global global_graph
    if global_graph is None:
        raise HTTPException(status_code=400, detail="No graph has been set.")

    json_data = DspJson(
        ir_version=1, producer_name="test", producer_version="0.1", graph=global_graph
    )
    pipeline = make_pipeline(json_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dsp_pipeline"
        pipeline.draw(tmp_path)  # writes "dsp_pipeline.svg" in that directory

        svg_file = tmp_path.with_suffix(".svg")
        if not svg_file.exists():
            return {"error": "SVG file could not be generated"}

        svg_content = svg_file.read_text()

    return Response(content=svg_content, media_type="image/svg+xml")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
