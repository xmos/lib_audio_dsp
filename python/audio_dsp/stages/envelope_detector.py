# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class EnvelopeDetectorPeak(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("envelope_detector_peak"), **kwargs)
        self.create_outputs(0)

        at = 0.01
        rt = 0.2
        self.dsp_block = drc.envelope_detector_peak(self.fs, self.n_in, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)

    def make_env_det_peak(self, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        """Update envelope detector configuration based on new parameters."""
        self.details = dict(
            attack_t=attack_t,
            release_t=release_t,
            delay=delay,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.envelope_detector_peak(
            self.fs, self.n_in, attack_t, release_t, delay, Q_sig
        )
        return self


class EnvelopeDetectorRMS(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("envelope_detector_rms"), **kwargs)
        self.create_outputs(0)

        at = 0.01
        rt = 0.2
        self.dsp_block = drc.envelope_detector_rms(self.fs, self.n_in, at, rt)

        self.set_control_field_cb("attack_alpha", lambda: self.dsp_block.attack_alpha_int)
        self.set_control_field_cb("release_alpha", lambda: self.dsp_block.release_alpha_int)

    def make_env_det_rms(self, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        """Update envelope detector configuration based on new parameters."""
        self.details = dict(
            attack_t=attack_t,
            release_t=release_t,
            delay=delay,
            Q_sig=Q_sig,
        )
        self.dsp_block = drc.envelope_detector_rms(
            self.fs, self.n_in, attack_t, release_t, delay, Q_sig
        )
        return self
