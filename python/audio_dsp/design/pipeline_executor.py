from pathlib import Path
from typing import Optional, NamedTuple
from collections.abc import Callable
from scipy.io import wavfile
from matplotlib import pyplot as plt

import numpy
import IPython


from .stage import Stage, StageOutput
from .graph import Graph
from ..dsp import signal_gen


class PipelineView(NamedTuple):
    stages: Optional[list[Stage]]
    inputs: list[StageOutput]
    outputs: list[StageOutput]


class ExecutionResult:
    """
    The result of processing samples through the pipeline.

    Parameters
    ----------
    result
        The data produced by the pipeline.
    fs
        sample rate

    Attributes
    ----------
    data
        ndarray containing the results of the pipeline.
    fs
        Sample rate.
    """

    def __init__(self, result: numpy.ndarray, fs: float):
        self.data = result
        self.fs = fs

    def to_wav(self, path: str | Path):
        """
        Save output to a wav file
        """
        wavfile.write(path, self.fs, self.data)

    def plot(self, path: Optional[str | Path] = None):
        """
        Display a time domain plot of the result. Save to file
        if path is not None.

        Parameters
        ----------
        path
            If path is not none then the plot will be saved to a file
            and not shown.
        """
        chans = self.data.shape[1]
        period = 1 / self.fs
        t = numpy.arange(0, self.data.shape[0] * period, period)
        fig, axs = plt.subplots(chans, sharex=True)
        for i in range(chans):
            axs[i].plot(t, self.data[:, i])
        if path:
            plt.savefig(path)
        else:
            plt.show()

    def play(self, channel: int):
        """
        Create a widget in the jupyter notebook to listen to the audio.

        .. warning::
            This will not work outside of a jupyter notebook.

        Parameters
        ----------
        channel
            The channel to listen to.
        """
        IPython.display.display(IPython.display.Audio(self.data[:, channel], rate=int(self.fs)))


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
        fs = i_edges[0].fs

        if data.shape[0] % frame_size:
            raise ValueError(f"Data length must be a multiple of frame size {frame_size}")

        ret = numpy.zeros((data.shape[0], n_o_chans))

        for index in range(0, data.shape[0], frame_size):
            inputs = [data[index : (index + frame_size), i] for i in range(n_i_chans)]
            outputs = self._process_frame(inputs, graph, i_edges, o_edges)
            for i, val in enumerate(outputs):
                ret[index : index + frame_size, i] = val
        return ExecutionResult(ret, fs)

    def log_chirp(
        self, length_s: float = 0.5, amplitude: float = 1, start: float = 20, stop: float = 20000
    ) -> ExecutionResult:
        """
        Generate a logarithmic chirp of constant amplitude and play through
        the simulated pipeline

        Parameters
        ----------
        length_s
            Length of generated chirp in seconds.
        amplitude
            Amplitude of the generated chirp, between 0 and 1.
        start
            Start frequency.
        stop
            Stop frequncy.

        Returns
        -------
        ExecutionResult
            The output wrapped in a helpful container for viewing, saving, processing, etc.
        """
        _, i_edges, _ = self._get_view()
        fs = i_edges[0].fs
        frame_size = i_edges[0].frame_size

        chirp = signal_gen.log_chirp(fs, length_s, amplitude, start=start, stop=stop)
        chirp_len = len(chirp)
        desired_chirp_len = chirp_len - (chirp_len % frame_size)
        chirp = chirp[:desired_chirp_len]
        chirp = numpy.stack([chirp] * len(i_edges), axis=1)

        return self.process(chirp)
