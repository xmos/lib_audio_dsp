# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path
from audio_dsp.design.stage import Stage
import numpy

class AddN(Stage):
    """
    Stage which adds a fixed constant to all inputs, for testing purposes
    """
    def __init__(self, n=0, **kwargs):
        super().__init__(config=Path(__file__).parent / "add_n.yaml", **kwargs)
        self.create_outputs(self.n_in)
        self["n"] = n

    def process(self, input: list[numpy.ndarray]):
        return [i + self["n"] for i in input]
