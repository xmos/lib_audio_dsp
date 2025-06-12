# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Noise suppressor and expander Stages control the behaviour
of quiet signals, typically by tring to reduce the audibility of noise
in the signal.
"""

from audio_dsp.design.stage import Stage, find_config
from audio_dsp.dsp import drc as drc
from audio_dsp.dsp import generic as dspg
from audio_dsp.models.noise_suppressor_expander_model import NoiseSuppressorExpanderParameters


class NoiseSuppressorExpander(Stage):
    """The Noise Suppressor (Expander) stage. A noise suppressor that
    reduces the level of an audio signal when it falls below a
    threshold. This is also known as an expander.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced relative to the expansion ratio over the
    release time. When the envelope returns above the threshold, the
    gain applied to the signal is increased to 1 over the attack time.

    The initial state of the noise suppressor is with the suppression
    off; this models a full scale signal having been present before
    t = 0.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.expander.noise_suppressor_expander`
        The DSP block class; see :ref:`NoiseSuppressorExpander`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("noise_suppressor_expander"), **kwargs)
        self.create_outputs(self.n_in)

        self.parameters = NoiseSuppressorExpanderParameters(
            threshold_db=-35, ratio=3, attack_t=0.005, release_t=0.12
        )
        self.set_parameters(self.parameters)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)
        self.set_control_field_cb("slope", lambda: self.dsp_block.slope_f32)

        self.stage_memory_parameters = (self.n_in,)

    def set_parameters(self, parameters: NoiseSuppressorExpanderParameters):
        """Update noise suppressor/expander configuration based on new parameters."""
        self.parameters = parameters
        self.dsp_block = drc.noise_suppressor_expander(
            self.fs,
            self.n_in,
            parameters.ratio,
            parameters.threshold_db,
            parameters.attack_t,
            parameters.release_t,
            dspg.Q_SIG,
        )

    def make_noise_suppressor_expander(
        self, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG
    ):
        """
        Update noise suppressor (expander) configuration based on new parameters.

        All parameters are passed to the constructor of :class:`audio_dsp.dsp.drc.noise_suppressor_expander`.

        Parameters
        ----------
        ratio : float
            The expansion ratio applied to the signal when the envelope
            falls below the threshold.
        threshold_db : float
            The threshold level in decibels below which the audio signal is
            attenuated.
        attack_t : float
            Attack time of the noise suppressor in seconds.
        release_t : float
            Release time of the noise suppressor in seconds.
        """
        parameters = NoiseSuppressorExpanderParameters(
            ratio=ratio, threshold_db=threshold_db, attack_t=attack_t, release_t=release_t
        )
        self.set_parameters(parameters)
