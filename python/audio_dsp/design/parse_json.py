import copy
import tempfile
from pathlib import Path
from typing import Annotated, Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import audio_dsp.stages as Stages
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.design.stage import StageOutputList, all_useable_stages, edgeProducerBaseModel

_stage_Models = Annotated[
    Union[tuple(i.Model for i in all_useable_stages().values())], Field(discriminator="op_type")
]


class Input(edgeProducerBaseModel):
    name: str
    output: list[int] = []
    channels: int
    fs: int


class Output(edgeProducerBaseModel):
    name: str
    input: list[int] = []
    channels: int
    fs: Optional[int] = None


class Graph(BaseModel):
    name: str
    nodes: list[_stage_Models]  # type: ignore
    input: Input
    output: Output


class DspJson(BaseModel):
    ir_version: int
    producer_name: str
    producer_version: str
    graph: Graph


def stage_handle(model):
    return getattr(Stages, model.op_type)


def make_pipeline(json_obj: DspJson) -> Pipeline:
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
    edge_list = [None] * (max(edgelist) + 1)
    for n in range(graph.input.channels):
        edge_list[n] = in_edges[n]

    waiting_nodes = list(range(len(graph.nodes)))

    while waiting_nodes:
        idx = waiting_nodes[0]
        this_node = graph.nodes[idx]

        # get node inputs
        stage_inputs = []
        for i in this_node.input:
            stage_inputs.append(edge_list[i])

        if None in stage_inputs:
            # input doesn't exist yet, try next node, add this node to the end
            waiting_nodes.pop(0)
            waiting_nodes.append(idx)
            continue

        stage_inputs = sum(stage_inputs, start=StageOutputList())
        node_output = p.stage(
            stage_handle(this_node),
            stage_inputs,
            this_node.name,
            thread=this_node.thread,
            **dict(this_node.config),
        )

        # set parameters if supported
        p.stages[-1].set_parameters(this_node.parameters)

        # if has outputs, add to edge to edge list
        if len(node_output) != 0:
            for i in range(len(this_node.output)):
                if edge_list[this_node.output[i]] is None:
                    edge_list[this_node.output[i]] = node_output[i]
                else:
                    raise ValueError("Edge conflict or multiple assignment occurred.")

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


@app.get("/schema")
def get_dsp_json_schema():
    """
    Return JSON schema for DspJson with the 'parameters' field
    stripped out of all models under _stage_Models.
    """
    schema = copy.deepcopy(DspJson.model_json_schema())
    return JSONResponse(schema)


@app.post("/render")
def render_dsp(json_data: DspJson):
    """Render the DSP pipeline diagram as an SVG image."""
    pipeline = make_pipeline(json_data)

    # Write the pipeline diagram to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dsp_pipeline"
        pipeline.draw(tmp_path)  # writes "dsp_pipeline.svg" in that directory

        svg_file = tmp_path.with_suffix(".svg")
        if not svg_file.exists():
            return {"error": "SVG file could not be generated"}

        # Read the SVG content
        svg_content = svg_file.read_text()

    # Return as image/svg+xml
    return Response(content=svg_content, media_type="image/svg+xml")


@app.get("/params")
def get_params(json_data: DspJson):
    """Return the JSON schema for the parameters of each stage."""
    nodes = json_data.graph.nodes
    params = {n.name: n.parameters.__class__.model_json_schema() for n in nodes}
    return JSONResponse(params)


# Uncomment this if you prefer to run with `python main.py`
# Otherwise you can run with `uvicorn main:app --reload`
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
