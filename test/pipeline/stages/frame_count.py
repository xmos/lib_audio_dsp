
# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from audio_dsp.design.stage import Stage

class FrameCount(Stage):
    """
    Stage which outputs a frame index
    """
    def __init__(self, **kwargs):
        super().__init__(name="frame_count", **kwargs)
        self.create_outputs(self.n_in)
