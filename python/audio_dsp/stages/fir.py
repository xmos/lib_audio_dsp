# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The FIR pipeline stage."""

from ..design.stage import Stage, find_config
from ..dsp import fir as fir
from ..dsp import generic as dspg


class FirDirect(Stage):
    """A FIR filter implemented in the time domain. The input signal is
    convolved with the filter coefficients. The filter coefficients can
    only be set at compile time.

    Attributes
    ----------
    dsp_block : None | audio_dsp.dsp.generic.dsp_block
        The dsp block class, see :class:`audio_dsp.dsp.fir.fir_direct`
        for implementation details.
    """

    def __init__(self, coeffs_path, **kwargs):
        super().__init__(name="fir_direct", **kwargs)

        self.create_outputs(self.n_in)

        self.dsp_block = fir.fir_direct(self.fs, self.n_in, coeffs_path)

        self.set_constant("n_taps", self.dsp_block.n_taps, "int32_t")
        self.set_constant("shift", self.dsp_block.shift, "int32_t")
        self.set_constant("coeffs", self.dsp_block.coeffs_int, "int32_t")

        self.stage_memory_parameters = (self.n_in, self.dsp_block.n_taps)

    def make_fir_direct(self, coeffs_path, Q_sig=dspg.Q_SIG):
        """Update fir configuration based on new parameters."""
        self.details = dict(
            coeffs_path=coeffs_path,
            Q_sig=Q_sig,
        )
        self.dsp_block = fir.fir_direct(self.fs, self.n_in, coeffs_path, Q_sig)
        return self
