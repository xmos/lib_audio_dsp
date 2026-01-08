# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from audio_dsp.design.stage import Stage
from pathlib import Path


class Dummy(Stage):
    """Stage that does nothing but has control"""
    def __init__(self, **kwargs):
        super().__init__(config=Path(__file__).parent / "dummy.yaml", **kwargs)
        self.create_outputs(self.n_in)
