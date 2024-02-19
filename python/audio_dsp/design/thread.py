# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from .composite_stage import CompositeStage
from .stage import Stage, find_config, ValueControlField
from .graph import Edge, Node
import yaml
from pathlib import Path


class DSPThreadStage(Stage):
    """
    Stage for the DSP thread. Does not support processing of data through it. Only
    used for DSP thread level control commands, for example, querying the max cycles
    consumed by the thread.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("dsp_thread"), **kwargs)
        self.create_outputs(0)

    """
    Override the CompositeStage.add_to_dot() function to ensure DSPThreadStage
    type stages are not added to the dot diagram

    Parameters
    ----------
    dot : graphviz.Diagraph
        dot instance to add edges to.
    """

    def add_to_dot(self, dot):  # Override this to not add the stage to the diagram
        return


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
        self.thread_stage = self.stage(DSPThreadStage, [])

    def __enter__(self):
        """Support for context manager"""
        return self

    def __exit__(self, type, value, traceback):
        """Support for context manager"""
        ...
