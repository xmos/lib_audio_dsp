# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Envelope detector Stages measure how the average or peak amplitude of
a signal varies over time.
"""

from audio_dsp.design.stage import Stage, find_config
from audio_dsp.dsp import drc as drc
from audio_dsp.dsp import generic as dspg
from audio_dsp.models.envelope_detector import EnvelopeDetectorParameters


class EnvelopeDetectorPeak(Stage):
    """
    A stage with no outputs that measures the signal peak envelope.

    The current envelope of the signal can be read out using this stage's
    ``envelope`` control.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.envelope_detector_peak`
        The DSP block class; see :ref:`EnvelopeDetectorPeak`
        for implementation details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("envelope_detector_peak"), **kwargs)
        self.create_outputs(0)

        self.parameters = EnvelopeDetectorParameters(
            attack_t=0.01,
            release_t=0.2,
        )
        self.set_parameters(self.parameters)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)

        self.stage_memory_parameters = (self.n_in,)

    def set_parameters(self, parameters: EnvelopeDetectorParameters):
        """Update the parameters of the EnvelopeDetectorPeak stage."""
        self.parameters = parameters
        self.dsp_block = drc.envelope_detector_peak(
            self.fs, self.n_in, parameters.attack_t, parameters.release_t, dspg.Q_SIG
        )

    def make_env_det_peak(self, attack_t, release_t, Q_sig=dspg.Q_SIG):
        """Update envelope detector configuration based on new parameters.

        Parameters
        ----------
        attack_t : float
            Attack time of the envelope detector in seconds.
        release_t : float
            Release time of the envelope detector in seconds.
        """
        parameters = EnvelopeDetectorParameters(attack_t=attack_t, release_t=release_t)
        self.set_parameters(parameters)


class EnvelopeDetectorRMS(Stage):
    """
    A stage with no outputs that measures the signal RMS envelope.

    The current envelope of the signal can be read out using this stage's
    ``envelope`` control.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.envelope_detector_rms`
        The DSP block class; see :ref:`EnvelopeDetectorRMS`
        for implementation details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("envelope_detector_rms"), **kwargs)
        self.create_outputs(0)

        self.parameters = EnvelopeDetectorParameters(
            attack_t=0.01,
            release_t=0.2,
        )
        self.set_parameters(self.parameters)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)

        self.stage_memory_parameters = (self.n_in,)

    def set_parameters(self, parameters: EnvelopeDetectorParameters):
        """Update the parameters of the EnvelopeDetectorRMS stage."""
        self.parameters = parameters
        self.dsp_block = drc.envelope_detector_rms(
            self.fs, self.n_in, parameters.attack_t, parameters.release_t, dspg.Q_SIG
        )

    def make_env_det_rms(self, attack_t, release_t, Q_sig=dspg.Q_SIG):
        """Update envelope detector configuration based on new parameters.

        Parameters
        ----------
        attack_t : float
            Attack time of the envelope detector in seconds.
        release_t : float
            Release time of the envelope detector in seconds.
        """
        parameters = EnvelopeDetectorParameters(attack_t=attack_t, release_t=release_t)
        self.set_parameters(parameters)
