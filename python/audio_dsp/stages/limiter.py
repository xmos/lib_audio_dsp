# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Assorted stages for limiting the signal."""

from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class LimiterRMS(Stage):
    """A limiter based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to it's original level
    after the envelope is below the threshold.

    Attributes
    ----------
    dsp_block : audio_dsp.dsp.drc.drc.limiter_rms
        The dsp block class, see
        :class:`audio_dsp.dsp.drc.drc.limiter_rms` for implementation
        details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_rms"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.limiter_rms(self.fs, self.n_in, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)

        self.stage_memory_parameters = (self.n_in,)

    def make_limiter_rms(self, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        """Update limiter configuration based on new parameters.

        Parameters
        ----------
        threshold_db : float
            Threshold in decibels above which limiting occurs.
        attack_t : float
            Attack time of the limiter in seconds.
        release_t : float
            Release time of the limiter in seconds.
        """
        self.details = dict(
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.limiter_rms(
            self.fs, self.n_in, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self


class LimiterPeak(Stage):
    """
    A limiter based on the peak value of the signal. When the peak
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to it's original level
    after the envelope is below the threshold.

    Attributes
    ----------
    dsp_block : audio_dsp.dsp.drc.drc.compressor_rms
        The dsp block class, see
        :class:`audio_dsp.dsp.drc.drc.limiter_peak` for implementation
        details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_peak"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.limiter_peak(self.fs, self.n_in, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)

        self.stage_memory_parameters = (self.n_in,)

    def make_limiter_peak(self, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        """Update limiter configuration based on new parameters.

        Parameters
        ----------
        threshold_db : float
            Threshold in decibels above which limiting occurs.
        attack_t : float
            Attack time of the limiter in seconds.
        release_t : float
            Release time of the limiter in seconds.

        """
        self.details = dict(
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.limiter_peak(
            self.fs, self.n_in, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self


class HardLimiterPeak(Stage):
    """
    A limiter based on the peak value of the signal, that never allows
    the signal to be higher than the threshold.

    When the peak envelope of the signal exceeds the threshold, the
    signal amplitude is reduced. If the signal still exceeds the
    threshold, it is clipped.

    The threshold set the value above which limiting/clipping occurs.
    The attack time sets how fast the limiter starts limiting. The
    release time sets how long the signal takes to ramp up to it's
    original level after the envelope is below the threshold.

    Attributes
    ----------
    dsp_block : audio_dsp.dsp.drc.drc.hard_limiter_peak
        The dsp block class, see
        :class:`audio_dsp.dsp.drc.drc.hard_limiter_peak` for
        implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("hard_limiter_peak"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.hard_limiter_peak(self.fs, self.n_in, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)

        self.stage_memory_parameters = (self.n_in,)

    def make_hard_limiter_peak(self, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        """Update limiter configuration based on new parameters.

        Parameters
        ----------
        threshold_db : float
            Threshold in decibels above which limiting occurs.
        attack_t : float
            Attack time of the limiter in seconds.
        release_t : float
            Release time of the limiter in seconds.
        """
        self.details = dict(
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.hard_limiter_peak(
            self.fs, self.n_in, threshold_db, attack_t, release_t, delay, Q_sig
        )
        return self


class Clipper(Stage):
    """
    A simple clipper that limits the signal to a specified threshold.

    If the signal is greater than the threshold level, it is set to the
    threshold value.

    Attributes
    ----------
    dsp_block : audio_dsp.dsp.drc.drc.clipper
        The dsp block class, see :class:`audio_dsp.dsp.drc.drc.clipper`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("clipper"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        self.dsp_block = drc.clipper(self.fs, self.n_in, threshold)

        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)

        self.stage_memory_parameters = (self.n_in,)

    def make_clipper(self, threshold_db, Q_sig=dspg.Q_SIG):
        """Update clipper configuration based on new parameters.

        Parameters
        ----------
        threshold_db : float
            Threshold in decibels above which clipping occurs.

        """
        self.details = dict(
            threshold_db=threshold_db,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.clipper(self.fs, self.n_in, threshold_db, Q_sig)
        return self
