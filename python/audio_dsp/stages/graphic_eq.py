# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Graphic EQs allow frequency response adjustments at fixed center
frequencies.
"""

from ..design.stage import Stage, find_config, StageOutputList, StageOutput
from ..dsp import generic as dspg
import audio_dsp.dsp.graphic_eq as geq
import numpy as np


class GraphicEq10b(Stage):
    """
    A 10 band graphic equaliser, with octave spaced center frequencies.
    The center frequencies are:
    [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]. The gain of 
    each band can be adjusted between -12 and + 12 dB.
    
    Note that for a 32 kHz sample rate, the 16 kHz band is not available,
    making a 9 band EQ. For a 16 kHz sample rate the 8k and 16 kHz bands
    are not available, making an 8 band EQ.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.graphic_eq.graphic_eq_10_band`
        The DSP block class; see :ref:`Graphic_Eq` for implementation details
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("graphic_eq_10b"), **kwargs)
        self.create_outputs(self.n_in)
        self.dsp_block = geq.graphic_eq_10_band(self.fs, self.n_in, np.zeros(10))
        self.set_control_field_cb("gains", lambda: self.dsp_block.gains_int)

        self.set_constant("coeffs", self.dsp_block._get_coeffs(), "int32_t")
        self.stage_memory_parameters = (self.n_in,)

    def set_gains(self, gains_db):
        """
        Set the gains of the graphic eq in dB.

        Parameters
        ----------
        gains_db : float
            The gains of the graphic eq in dB.
        """
        self.dsp_block.gains_db = gains_db
        return self