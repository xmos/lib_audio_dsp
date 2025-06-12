# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Limiter Stages allow the amplitude of the signal to be restricted
based on its envelope.
"""

from audio_dsp.design.stage import Stage, find_config
from audio_dsp.dsp import drc as drc
from audio_dsp.dsp import generic as dspg
from audio_dsp.models.limiter_model import LimiterParameters, ClipperParameters


class LimiterRMS(Stage):
    """A limiter based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold sets the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to its original level
    after the envelope is below the threshold.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.limiter_rms`
        The DSP block class; see :ref:`LimiterRMS`
        for implementation details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_rms"), **kwargs)
        self.create_outputs(self.n_in)

        self.parameters = LimiterParameters(threshold_db=0, attack_t=0.01, release_t=0.2)
        self.set_parameters(self.parameters)

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
        parameters = LimiterParameters(
            threshold_db=threshold_db, attack_t=attack_t, release_t=release_t
        )
        self.set_parameters(parameters)

    def set_parameters(self, parameters: LimiterParameters):
        """Update limiter configuration based on new parameters.

        Parameters
        ----------
        parameters : LimiterParameters
            The parameters to update the limiter with.
        """
        self.parameters = parameters
        self.dsp_block = drc.limiter_rms(
            self.fs,
            self.n_in,
            parameters.threshold_db,
            parameters.attack_t,
            parameters.release_t,
            dspg.Q_SIG,
        )


class LimiterPeak(Stage):
    """
    A limiter based on the peak value of the signal. When the peak
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold sets the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to its original level
    after the envelope is below the threshold.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.limiter_peak`
        The DSP block class; see :ref:`LimiterPeak`
        for implementation details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_peak"), **kwargs)
        self.create_outputs(self.n_in)

        self.parameters = LimiterParameters(threshold_db=0, attack_t=0.01, release_t=0.2)
        self.set_parameters(self.parameters)

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
        parameters = LimiterParameters(
            threshold_db=threshold_db, attack_t=attack_t, release_t=release_t
        )
        self.set_parameters(parameters)

    def set_parameters(self, parameters: LimiterParameters):
        """Update limiter configuration based on new parameters.

        Parameters
        ----------
        parameters : LimiterParameters
            The parameters to update the limiter with.
        """
        self.parameters = parameters
        self.dsp_block = drc.limiter_peak(
            self.fs,
            self.n_in,
            parameters.threshold_db,
            parameters.attack_t,
            parameters.release_t,
            dspg.Q_SIG,
        )


class HardLimiterPeak(Stage):
    """
    A limiter based on the peak value of the signal. The peak
    envelope of the signal may never exceed the threshold.

    When the peak envelope of the signal exceeds the threshold, the
    signal amplitude is reduced. If the signal still exceeds the
    threshold, it is clipped.

    The threshold sets the value above which limiting/clipping occurs.
    The attack time sets how fast the limiter starts limiting. The
    release time sets how long the signal takes to ramp up to its
    original level after the envelope is below the threshold.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.hard_limiter_peak`
        The DSP block class; see :ref:`HardLimiterPeak` for
        implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("hard_limiter_peak"), **kwargs)
        self.create_outputs(self.n_in)

        self.parameters = LimiterParameters(threshold_db=0, attack_t=0.01, release_t=0.2)
        self.set_parameters(self.parameters)

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
        parameters = LimiterParameters(
            threshold_db=threshold_db, attack_t=attack_t, release_t=release_t
        )
        self.set_parameters(parameters)

    def set_parameters(self, parameters: LimiterParameters):
        """Update limiter configuration based on new parameters.

        Parameters
        ----------
        parameters : LimiterParameters
            The parameters to update the limiter with.
        """
        self.parameters = parameters
        self.dsp_block = drc.hard_limiter_peak(
            self.fs,
            self.n_in,
            parameters.threshold_db,
            parameters.attack_t,
            parameters.release_t,
            dspg.Q_SIG,
        )


class Clipper(Stage):
    """
    A simple clipper that limits the signal to a specified threshold.

    If the signal is greater than the threshold level, it is set to the
    threshold value.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.clipper`
        The DSP block class; see :ref:`Clipper`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("clipper"), **kwargs)
        self.create_outputs(self.n_in)

        self.parameters = ClipperParameters(threshold_db=0)
        self.set_parameters(self.parameters)

        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)

        self.stage_memory_parameters = (self.n_in,)

    def make_clipper(self, threshold_db, Q_sig=dspg.Q_SIG):
        """Update clipper configuration based on new parameters.

        Parameters
        ----------
        threshold_db : float
            Threshold in decibels above which clipping occurs.

        """
        parameters = ClipperParameters(threshold_db=threshold_db)
        self.set_parameters(parameters)

    def set_parameters(self, parameters: ClipperParameters):
        """Update clipper configuration based on new parameters.

        Parameters
        ----------
        parameters : LimiterParameters
            The parameters to update the clipper with.
        """
        self.parameters = parameters
        self.dsp_block = drc.clipper(self.fs, self.n_in, parameters.threshold_db, dspg.Q_SIG)
