# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Some words about graphic EQ.
"""

from ..design.stage import Stage, find_config, StageOutputList, StageOutput
from ..dsp import generic as dspg
import audio_dsp.dsp.graphic_eq as geq
import numpy as np


class GraphicEq10b(Stage):
    """
    Mixes the input signals together. The mixer can be used to add signals
    together, or to attenuate the input signals.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.mixer`
        The DSP block class; see :ref:`Mixer` for implementation details
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("graphic_eq_10b"), **kwargs)
        self.create_outputs(1)
        self.dsp_block = geq.graphic_eq_10_band(self.fs, self.n_in, np.zeros(10))
        self.set_control_field_cb("gains", lambda: self.dsp_block.gains_int)
        self.set_constant("sampling_freq", self.fs, "int32_t")

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