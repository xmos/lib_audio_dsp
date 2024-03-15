# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class NoiseGate(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("noise_gate"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = -35
        at = 0.005
        rt = 0.12
        self.dsp_block = drc.noise_gate(self.fs, self.n_in, threshold, at, rt)

        # Swapping alphas as they are the other way around in python
        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.release_alpha_f32)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.attack_alpha_f32)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_f32)

    def make_noise_gate(self, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        """Update noise gate configuration based on new parameters."""
        self.details = dict(
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            delay=delay,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.noise_gate(
            self.fs, self.n_in, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self