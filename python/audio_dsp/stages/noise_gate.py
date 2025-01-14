# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Noise gate Stages remove quiet signals from the audio output."""

from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class NoiseGate(Stage):
    """A noise gate that reduces the level of an audio signal when it
    falls below a threshold.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced to 0 over the release time. When the
    envelope returns above the threshold, the gain applied to the signal
    is increased to 1 over the attack time.

    The initial state of the noise gate is with the gate open (no
    attenuation); this models a full scale signal having been present before
    t = 0.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.expander.noise_gate`
        The DSP block class; see :ref:`NoiseGate` for implementation
        details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("noise_gate"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = -35
        at = 0.005
        rt = 0.12
        self.dsp_block = drc.noise_gate(self.fs, self.n_in, threshold, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)
        self.set_control_field_cb("threshold", lambda: self.dsp_block.threshold_int)

        self.stage_memory_parameters = (self.n_in,)

    def make_noise_gate(self, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        """Update noise gate configuration based on new parameters.

        Parameters
        ----------
        threshold_db : float
            The threshold level in decibels below which the audio signal is
            attenuated.
        attack_t : float
            Attack time of the noise gate in seconds.
        release_t : float
            Release time of the noise gate in seconds.
        """
        self.details = dict(
            threshold_db=threshold_db,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.noise_gate(
            self.fs, self.n_in, threshold_db, attack_t, release_t, Q_sig
        )
        return self
