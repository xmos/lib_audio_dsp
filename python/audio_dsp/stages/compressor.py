# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class CompressorRMS(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("compressor_rms"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        ratio = 4
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.compressor_rms(self.fs, self.n_in, ratio, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)
        self.set_control_field_cb("slope", lambda: self.dsp_block.slope_f32)

        self.stage_memory_string = "compressor_rms"
        self.stage_memory_parameters = (self.n_in,)

    def make_compressor_rms(
        self, ratio, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG
    ):
        """Update compressor configuration based on new parameters."""
        self.details = dict(
            ratio=ratio,
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            delay=delay,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.compressor_rms(
            self.fs, self.n_in, ratio, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self
