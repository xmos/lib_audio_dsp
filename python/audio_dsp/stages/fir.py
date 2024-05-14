# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The compressor stage."""

from ..design.stage import Stage, find_config
from ..dsp import fir as fir
from ..dsp import generic as dspg


class Fir_Direct(Stage):
    """A FIR filter implemented in the time domain

    See :class:`audio_dsp.dsp.fir.fir_direct` for details.
    """

    def __init__(self, coeffs_path, **kwargs):
        super().__init__(config=find_config("fir_direct"), **kwargs)
        self.create_outputs(self.n_in)

        self.dsp_block = fir.fir_direct(self.fs, self.n_in, coeffs_path)

        self.set_control_field_cb("n_taps", lambda: self.dsp_block.n_taps)
        self.set_control_field_cb("shift", lambda: self.dsp_block.shift)
        self.set_control_field_cb("coeffs", lambda: self.dsp_block.coeffs_int)

        self.stage_memory_parameters = (self.n_in, self.dsp_block.n_taps)

    def make_fir_direct(
        self, coeffs_path, coeff_scaling=None, Q_sig=dspg.Q_SIG
    ):
        """Update fir configuration based on new parameters."""
        self.details = dict(
            coeffs_path-coeffs_path,
            coeff_scaling=coeff_scaling,
            Q_sig=Q_sig,
        )
        self.dsp_block = fir.fir_direct(
            self.fs, self.n_in, coeffs_path, coeff_scaling, Q_sig
        )
        return self
