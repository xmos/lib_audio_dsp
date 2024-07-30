# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Compressor stages allow for control of the dynamic range of the
signal, such as reducing the level of loud sounds.
"""

from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class CompressorRMS(Stage):
    """A compressor based on the RMS envelope of the input signal.

    When the RMS envelope of the signal exceeds the threshold, the
    signal amplitude is reduced by the compression ratio.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the compressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to its original level after the envelope is below the
    threshold.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.compressor_rms`
        The DSB block class; see :ref:`CompressorRMS`
        for implementation details.
    """

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

        self.stage_memory_parameters = (self.n_in,)

    def make_compressor_rms(self, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
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
        self.dsp_block = drc.compressor_rms(
            self.fs, self.n_in, ratio, threshold_db, attack_t, release_t, Q_sig
        )
        return self
