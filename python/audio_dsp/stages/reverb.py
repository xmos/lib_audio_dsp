# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
import audio_dsp.dsp.reverb as rvrb
import numpy as np

class Reverb(Stage):
    def __init__(self, max_room_size=1, **kwargs):
        super().__init__(config=find_config("reverb"), **kwargs)
        if self.fs is None:
            raise ValueError("Reverb requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)

        decay = 0.5
        wet_gain_db = -1.0
        dry_gain_db = -1.0
        self.reverb = rvrb.reverb_room(self.fs, self.n_in, max_room_size=max_room_size, decay=decay, wet_gain_db=wet_gain_db, dry_gain_db=dry_gain_db)
        self["sampling_freq"] = self.fs
        self["max_room_size"] = float(max_room_size)
        self.set_control_field_cb("room_size", lambda: self.reverb.room_size)
        self.set_control_field_cb("damping", lambda: self.reverb.damping)
        self.set_control_field_cb("pregain", lambda: self.reverb.pregain)
        self["decay"] = decay
        self["wet_gain_db"] = wet_gain_db
        self["dry_gain_db"] = dry_gain_db

