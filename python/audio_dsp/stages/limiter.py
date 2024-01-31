from ..design.stage import Stage, find_config
from ..dsp import drc as drc
import numpy as np


class LimiterPeak(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_peak"), **kwargs)
        self.create_outputs(self.n_in)

        fs = 48000
        threshold = 0
        at = 0.01
        rt = 0.2
        self.lt = drc.limiter_peak(fs, threshold, at, rt)

        self.set_control_field_cb("attack_alpha",
                                  lambda: self.lt.attack_alpha_uq30)
        self.set_control_field_cb("release_alpha",
                                  lambda: self.lt.release_alpha_uq30)
        self.set_control_field_cb("threshold",
                                  lambda: [self.lt.threshold_s32.mant, self.lt.threshold_s32.exp])

    def process(self, in_channels):
        """
        Run Biquad on the input channels and return the output

        Args:
            in_channels: list of numpy arrays

        Returns:
            list of numpy arrays.
        """
