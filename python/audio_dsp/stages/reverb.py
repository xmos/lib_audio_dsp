# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
import audio_dsp.dsp.reverb as rvrb
import numpy as np

class Reverb(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("reverb"), **kwargs)
        if self.fs is None:
            raise ValueError("Reverb requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.reverb = rvrb.reverb_room(self.fs, self.n_in)
