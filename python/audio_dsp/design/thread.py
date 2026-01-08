# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Contains classes for adding a thread to the DSP pipeline."""

from typing import Type

from audio_dsp.design.composite_stage import CompositeStage, _StageOrComposite
from audio_dsp.design.stage import Stage, StageOutputList, find_config


class DSPThreadStage(Stage):
    """
    Stage for the DSP thread. Does not support processing of data through it. Only
    used for DSP thread level control commands, for example, querying the max cycles
    consumed by the thread.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("dsp_thread"), **kwargs)
        self.create_outputs(0)

    def add_to_dot(self, dot):
        """
        Exclude this stage from the dot diagram.

        Parameters
        ----------
        dot : graphviz.Diagraph
            dot instance to add edges to.
        """


class Thread(CompositeStage):
    """
    A composite stage used to represent a thread in the pipeline. Create
    using Pipeline.thread rather than instantiating directly.

    Parameters
    ----------
    id : int
        Thread index
    kwargs : dict
        forwarded to __init__ of CompositeStage

    Attributes
    ----------
    id : int
        Thread index
    thread_stage : Stage
        DSPThreadStage stage
    """

    def __init__(self, id: int, **kwargs):
        super().__init__(name=f"Thread {id}", **kwargs)
        self.id = id
        self.thread_stage = None

    def __enter__(self):
        """Support for context manager."""
        return self

    def __exit__(self, type, value, traceback):
        """Support for context manager."""
        ...

    def add_thread_stage(self):
        """Add to this thread the stage which manages thread level commands."""
        self.thread_stage = self.stage(DSPThreadStage, StageOutputList(), label=f"thread{self.id}")

    def stage(
        self,
        stage_type: Type[_StageOrComposite],
        inputs: StageOutputList,
        label: str | None = None,
        **kwargs,
    ) -> _StageOrComposite:
        """
        Create a new stage or composite stage and
        register it with this composite stage.

        This is a wrapper around :func:`audio_dsp.design.composite_stage.stage`
        but adds a thread crossing counter.
        """
        for i in inputs.edges:
            if i:
                i.crossings.append(self.id)
        stage = super().stage(stage_type, inputs, label, **kwargs)
        return stage
