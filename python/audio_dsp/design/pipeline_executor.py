from typing import Optional, NamedTuple
from collections.abc import Callable

import numpy


from .stage import Stage, StageOutput
from .graph import Graph


class PipelineView(NamedTuple):
    stages: Optional[list[Stage]]
    inputs: list[StageOutput]
    outputs: list[StageOutput]


class PipelineExecutor:
    """
    Utility for simulating the pipeline

    Parameters
    ----------
    graph
        The pipeline graph to simulate
    """

    def __init__(self, graph: Graph[Stage], view_getter: Callable[[], PipelineView]):
        self._graph = graph
        self._view_getter = view_getter

    def _get_view(self) -> tuple[Graph[Stage], list[StageOutput], list[StageOutput]]:
        view = self._view_getter()
        graph = self._graph.get_view(view.stages) if view.stages else self._graph
        return graph, view.inputs, view.outputs

    def _process_frame(
        self,
        frame: list[numpy.ndarray],
        graph: Graph[Stage],
        i_edges: list[StageOutput],
        o_edges: list[StageOutput],
    ) -> list[numpy.ndarray]:
        """
        process channels through the pipeline and return the result.
        """
        edges = {}
        for edge, data in zip(i_edges, frame):
            edges[edge] = data

        stages = graph.sort()

        for stage in stages:
            inputs = [edges[e] for e in stage.i]
            if stage.o or stage.i:
                outputs = stage.process(inputs)
                edges.update({e: o for e, o in zip(stage.o, outputs)})

        return [edges[e] for e in o_edges]

    def process(self, data: numpy.ndarray):
        #
        # TODO
        #  - Convert data to a float between -1 and 1, assume int inputs range from INT_MIN to INT_MAX
        #  - the dsp_block methods all expect float
        #  - See test_stages.py for example of how the data io should look.
        #  - Not sure if test should be bit exact with C... maybe both are needed.
        #
        #
        graph, i_edges, o_edges = self._get_view()
        n_i_chans = len(i_edges)
        n_o_chans = len(o_edges)

        if len(data.shape) == 1:
            data = data.reshape((data.shape[0], 1))
        if len(data.shape) != 2:
            raise ValueError("Can only process 2D or 1D arrays of inputs")
        if data.shape[1] != n_i_chans:
            raise ValueError(f"Received {data.shape[1]} channels, expected {n_i_chans}")

        # Assume all edges have the same frame size
        frame_size = i_edges[0].frame_size

        if data.shape[0] % frame_size:
            raise ValueError(f"Data length must be a multiple of frame size {frame_size}")

        ret = numpy.zeros((data.shape[0], n_o_chans))

        for index in range(0, data.shape[0], frame_size):
            inputs = [data[index : (index + frame_size), i] for i in range(n_i_chans)]
            outputs = self._process_frame(inputs, graph, i_edges, o_edges)
            for i, val in enumerate(outputs):
                ret[index : index + frame_size, i] = val
        return ret
