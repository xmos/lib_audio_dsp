# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Stage implementation for the noise suppressor (expander)."""

from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class NoiseSuppressorExpander(Stage):
    """The Noise Suppressor (Expander) stage.

    Implementation details can be found at :class:`audio_dsp.dsp.drc.noise_suppressor_expander`.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("noise_suppressor_expander"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = -35
        ratio = 3
        at = 0.005
        rt = 0.120
        self.dsp_block = drc.noise_suppressor_expander(
            self.fs, self.n_in, ratio, threshold, at, rt
        )

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)
        self.set_control_field_cb("slope", lambda: self.dsp_block.slope_f32)

        self.stage_memory_parameters = (self.n_in,)

    def make_noise_suppressor_expander(
        self, ratio, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG
    ):
        """
        Update noise suppressor (expander) configuration based on new parameters.

        All parameters are passed to the constructor of :class:`audio_dsp.dsp.drc.noise_suppressor_expander`.
        """
        self.details = dict(
            ratio=ratio,
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            delay=delay,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.noise_suppressor_expander(
            self.fs, self.n_in, ratio, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self
