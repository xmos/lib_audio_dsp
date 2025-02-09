# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Sidechain compressor Stages use the envelope of one input to control
the level of a different input.
"""

from audio_dsp.design.stage import Stage, find_config
from audio_dsp.dsp import drc as drc
from audio_dsp.dsp import generic as dspg
from audio_dsp.models.compressor_sidechain_model import CompressorSidechainParameters


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
    dsp_block : :class:`audio_dsp.dsp.drc.drc.compressor_sidechain`
        The DSP block class; see :ref:`CompressorSidechain`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("compressor_sidechain"), **kwargs)
        self.create_outputs(1)

        ratio = 3
        threshold = -35
        at = 0.005
        rt = 0.120
        self.dsp_block = drc.compressor_sidechain(self.fs, ratio, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)
        self.set_control_field_cb("slope", lambda: self.dsp_block.slope_f32)

        self.stage_memory_parameters = (2,)

    def set_parameters(self, parameters: CompressorSidechainParameters):
        self.make_compressor_sidechain(
            parameters.ratio, parameters.threshold_db, parameters.attack_t, parameters.release_t
        )

    def make_compressor_sidechain(
        self, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG
    ):
        """
        Update compressor configuration based on new parameters.

        All parameters are passed to the constructor of :class:`audio_dsp.dsp.drc.compressor_sidechain`.

        Parameters
        ----------
        ratio : float
            The compression ratio applied to the signal when the envelope
            exceeds the threshold.
        threshold_db : float
            The threshold level in decibels above which the audio signal is
            compressed.
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
        self.dsp_block = drc.compressor_sidechain(
            self.fs, ratio, threshold_db, attack_t, release_t, Q_sig
        )
        return self
