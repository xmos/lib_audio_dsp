# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class NoiseSuppressor(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("noise_suppressor"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = -35
        ratio = 3
        at = 0.005
        rt = 0.120
        self.dsp_block = drc.noise_suppressor(self.fs, self.n_in, ratio, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)
        self.set_control_field_cb("slope", lambda: self.dsp_block.slope_f32)

    def make_noise_suppressor(
        self, ratio, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG
    ):
        """Update noise suppressor configuration based on new parameters."""
        self.details = dict(
            ratio=ratio,
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            delay=delay,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.noise_suppressor(
            self.fs, self.n_in, ratio, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self