# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Utilities for processing the pipeline on the host machine."""

from pathlib import Path
from typing import Optional, NamedTuple
from collections.abc import Callable
from scipy.io import wavfile
from matplotlib import pyplot as plt

import numpy
from IPython import display


from .stage import Stage, StageOutput
from .graph import Graph
from ..dsp import signal_gen


class PipelineView(NamedTuple):
    """A view of the DSP pipeline that is used by PipelineExecutor."""

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
        """Save output to a wav file."""
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
        fig, axs = plt.subplots(chans, sharex=True, squeeze=False)
        for i in range(chans):
            axs[i][0].plot(t, self.data[:, i])
        if path:
            plt.savefig(path)
        else:
            plt.show()

    def plot_magnitude_spectrum(self, path: Optional[str | Path] = None):
        """
        Display a spectrum plot of the result. Save to file
        if path is not None.

        Parameters
        ----------
        path
            If path is not none then the plot will be saved to a file
            and not shown.
        """
        chans = self.data.shape[1]
        fig, axs = plt.subplots(chans, sharex=True, squeeze=False)
        for i in range(chans):
            axs[i][0].magnitude_spectrum(self.data[:, i], Fs=self.fs, scale="dB")
        if path:
            plt.savefig(path)
        else:
            plt.show()

    def plot_spectrogram(self, path: Optional[str | Path] = None):
        """
        Display a spectrogram plot of the result. Save to file
        if path is not None.

        Parameters
        ----------
        path
            If path is not none then the plot will be saved to a file
            and not shown.
        """
        chans = self.data.shape[1]
        fig, axs = plt.subplots(chans, sharex=True, squeeze=False)
        for i in range(chans):
            Pxx, _, _, _ = axs[i][0].specgram(self.data[:, i], NFFT=1024, Fs=self.fs, noverlap=900)
            Pxx_max_dB = 10 * numpy.log10(Pxx.max()) - 5
            axs[i][0].specgram(
                self.data[:, i],
                NFFT=1024,
                Fs=self.fs,
                noverlap=900,
                cmap=plt.get_cmap("cividis"),
                vmin=Pxx_max_dB - 50,
                vmax=Pxx_max_dB,
            )
        fig.supylabel("Frequency [Hz]")
        fig.supxlabel("Time [sec]")
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
        display.display(display.Audio(self.data[:, channel], rate=int(self.fs)))


class PipelineExecutor:
    """
    Utility for simulating the pipeline.

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
        """Process channels through the pipeline and return the result."""
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

    def process(self, data: numpy.ndarray) -> ExecutionResult:
        """
        Process the dsp pipeline on the host.

        Parameters
        ----------
        data
            Pipeline input to process through the pipeline. The shape must match the number of channels
            that the pipeline expects as an input. If this is 1 then it may be a 1 dimensional array. Otherwise
            it must have shape (num_samples, num_channels).

        Returns
        -------
        ExecutionResult
            A result object that can be used to visualise or save the output.
        """
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
        self,
        length_s: float = 0.5,
        amplitude: float = 1,
        start: float = 20,
        stop: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Generate a logarithmic chirp of constant amplitude and play through
        the simulated pipeline.

        Parameters
        ----------
        length_s
            Length of generated chirp in seconds.
        amplitude
            Amplitude of the generated chirp, between 0 and 1.
        start
            Start frequency.
        stop
            Stop frequency. Nyquist if not set

        Returns
        -------
        ExecutionResult
            The output wrapped in a helpful container for viewing, saving, processing, etc.
        """
        _, i_edges, _ = self._get_view()
        fs = i_edges[0].fs
        if fs is None:
            raise RuntimeError(
                "Executor makes the assumption that the pipeline edges all have fs set"
            )
        frame_size = i_edges[0].frame_size

        chirp = signal_gen.log_chirp(fs, length_s, amplitude, start=start, stop=stop or fs / 2)
        chirp_len = len(chirp)
        desired_chirp_len = chirp_len - (chirp_len % frame_size)
        chirp = chirp[:desired_chirp_len]
        chirp = numpy.stack([chirp] * len(i_edges), axis=1)

        return self.process(chirp)
