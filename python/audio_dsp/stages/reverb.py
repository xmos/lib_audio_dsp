# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The reverb stage."""

from ..design.stage import Stage, find_config
import audio_dsp.dsp.reverb as rvrb


class ReverbRoom(Stage):
    """
    The reverb stage.

    See :class:`audio_dsp.dsp.reverb.reverb_room` for details on implementation.

    Parameters
    ----------
    max_room_size
        Sets the maximum room size for this reverb. The ``room_size`` parameter sets the
        fraction of this value actually used at any given time.
    """

    def __init__(self, max_room_size=1, **kwargs):
        super().__init__(config=find_config("reverb_room"), **kwargs)
        if self.fs is None:
            raise ValueError("Reverb requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)

        self.dsp_block = rvrb.reverb_room(self.fs, self.n_in, max_room_size=max_room_size)
        self["sampling_freq"] = self.fs
        self["max_room_size"] = float(max_room_size)
        self.set_control_field_cb("room_size", lambda: self.dsp_block.room_size)
        self.set_control_field_cb("feedback", lambda: self.dsp_block.combs[0].feedback_int)
        self.set_control_field_cb("damping", lambda: self.dsp_block.combs[0].damp1_int)
        self.set_control_field_cb("wet_gain", lambda: self.dsp_block.wet_int)
        self.set_control_field_cb("pregain", lambda: self.dsp_block.pregain_int)
        self.set_control_field_cb("dry_gain", lambda: self.dsp_block.dry_int)

        self.stage_memory_parameters = (self["sampling_freq"], self["max_room_size"])

    def set_wet_gain(self, gain_dB):
        """
        Set the wet gain of the reverb room stage.

        Parameters
        ----------
        gain_db : float
            Wet gain in dB, less than 0 dB.
        """
        self.dsp_block.set_wet_gain(gain_dB)

    def set_dry_gain(self, gain_dB):
        """
        Set the dry gain of the reverb room stage.

        Parameters
        ----------
        gain_db : float
            Dry gain in dB, less than 0 dB.
        """
        self.dsp_block.set_dry_gain(gain_dB)

    def set_pre_gain(self, pre_gain):
        """
        Set the pre gain of the reverb room stage.

        Parameters
        ----------
        pre_gain : float
            Pre gain value, less than 1.
        """
        self.dsp_block.set_pre_gain(pre_gain)

    def set_room_size(self, new_room_size):
        """
        Set the room size, will adjust the delay line lengths.

        Parameters
        ----------
        new_room_size : float
            How big the room is as a proportion of max_room_size. This
            sets delay line lengths and must be between 0 and 1.
        """
        self.dsp_block.set_room_size(new_room_size)

    def set_damping(self, damping):
        """
        Set the damping of the reverb room stage.

        Parameters
        ----------
        damping : float
            How much high frequency attenuation in the room, between 0 and 1.
        """
        self.dsp_block.set_damping(damping)

    def set_decay(self, decay):
        """
        Set the decay of the reverb room stage.

        Parameters
        ----------
        decay : float
            How long the reverberation of the room is, between 0 and 1.
        """
        self.dsp_block.set_decay(decay)
