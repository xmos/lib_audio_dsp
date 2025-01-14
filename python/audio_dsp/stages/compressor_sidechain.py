# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Sidechain compressor Stages use the envelope of one input to control
the level of a different input.
"""

from ..design.stage import Stage, find_config, StageParameters
from ..dsp import drc as drc
from ..dsp import generic as dspg
from typing import Literal

from pydantic import Field


class CompressorSidechainParameters(StageParameters):
    ratio: float = Field(default=3)
    threshold_db: float = Field(default=-35)
    attack_t: float = Field(default=0.005)
    release_t: float = Field(default=0.120)


class CompressorSidechain(Stage):
    """
    An sidechain compressor based on the RMS envelope of the detect
    signal.

    This stage is limited to accepting 2 channels. The first is the channel that
    will be compressed. The second is the detect channel. The level of compression
    depends on the envelope of the second channel.

    When the RMS envelope of the detect signal exceeds the threshold, the
    processed signal amplitude is reduced by the compression ratio.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the compressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to its original level after the envelope is below the
    threshold.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.sidechain.compressor_rms_sidechain_mono`
        The DSP block class; see :ref:`CompressorSidechain`
        for implementation details.
    """

    class Model(Stage.Model):
        op_type: Literal["CompressorSidechain"] = "CompressorSidechain"
        parameters: CompressorSidechainParameters = Field(
            default_factory=CompressorSidechainParameters
        )

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

        self.stage_memory_parameters = (self.n_in,)

    def set_parameters(self, parameters: CompressorSidechainParameters):
        self.make_compressor_sidechain(
            parameters.ratio, parameters.threshold_db, parameters.attack_t, parameters.release_t
        )

    def make_compressor_sidechain(
        self, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG
    ):
        """Update compressor configuration based on new parameters.

        Parameters
        ----------
        ratio : float
            Compression gain ratio applied when the signal is above the
            threshold.
        threshold_db : float
            Threshold in decibels above which compression occurs.
        attack_t : float
            Attack time of the compressor in seconds.
        release_t : float
            Release time of the compressor in seconds.
        """
        self.details = dict(
            ratio=ratio,
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.compressor_rms_sidechain_mono(
            self.fs, ratio, threshold_db, attack_t, release_t, Q_sig
        )
        return self
