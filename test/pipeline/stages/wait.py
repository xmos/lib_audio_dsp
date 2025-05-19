# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from audio_dsp.design.stage import Stage

def _calc_ticks(ratio, fs, frame_size):
    frame_period = frame_size / fs
    return int(100e6 * ratio * frame_period)

class Wait(Stage):
    """
    Stage which adds a fixed constant to all inputs, for testing purposes
    """
    def __init__(self, wait_ratio=0.1, **kwargs):
        super().__init__(name="wait", **kwargs)
        self.create_outputs(self.n_in)
        self.set_constant("ticks",
                          _calc_ticks(wait_ratio, self.fs, self.frame_size),
                          "int32_t")

