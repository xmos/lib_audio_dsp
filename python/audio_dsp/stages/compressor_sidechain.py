# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The sidechain compressor."""

from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class CompressorSidechain(Stage):
    """
    A sidechain compressor.

    This stage is limited to accepting 2 channels. The first is the channel that
    will be compressed. The second is the detect channel. The level of compression
    depends on the envelope of the second channel.

    See :class:`audio_dsp.dsp.drc.compressor_rms_sidechain_mono` for details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("compressor_sidechain"), **kwargs)
        self.create_outputs(1)
        if self.n_in != 2:
            raise ValueError(f"Sidechain compressor requires 2 inputs, got {self.n_in}")

        threshold = 0
        ratio = 4
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.compressor_rms_sidechain_mono(self.fs, ratio, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)
        self.set_control_field_cb("slope", lambda: self.dsp_block.slope_f32)

    def make_compressor_sidechain(
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
        self.dsp_block = drc.compressor_rms_sidechain_mono(
            self.fs, ratio, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self
