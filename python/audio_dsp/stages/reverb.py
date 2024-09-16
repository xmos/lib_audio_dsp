# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Reverb Stages emulate the natural reverberance of rooms."""

from ..design.stage import Stage, find_config
import audio_dsp.dsp.reverb as rvrb


class ReverbRoom(Stage):
    """
    The room reverb stage. This is based on Freeverb by Jezar at
    Dreampoint, and consists of 8 parallel comb filters fed into 4
    series all-pass filters.

    Parameters
    ----------
    max_room_size
        Sets the maximum room size for this reverb. The ``room_size``
        parameter sets the fraction of this value actually used at any
        given time. For optimal memory usage, max_room_size should be
        set so that the longest reverb tail occurs when
        ``room_size=1.0``.
    predelay : float, optional
        The delay applied to the wet channel in ms.
    max_predelay : float, optional
        The maximum predelay in ms.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.reverb.reverb_room`
        The DSP block class; see :ref:`ReverbRoom`
        for implementation details.
    """

    def __init__(self, max_room_size=1, predelay=10, max_predelay=None, **kwargs):
        super().__init__(config=find_config("reverb_room"), **kwargs)
        if self.fs is None:
            raise ValueError("Reverb requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)

        max_predelay = predelay if max_predelay == None else max_predelay

        self.dsp_block = rvrb.reverb_room(self.fs, self.n_in, max_room_size=max_room_size, predelay=predelay, max_predelay=max_predelay)
        self.set_control_field_cb("room_size", lambda: self.dsp_block.room_size)
        self.set_control_field_cb("feedback", lambda: self.dsp_block.combs[0].feedback_int)
        self.set_control_field_cb("damping", lambda: self.dsp_block.combs[0].damp1_int)
        self.set_control_field_cb("wet_gain", lambda: self.dsp_block.wet_int)
        self.set_control_field_cb("pregain", lambda: self.dsp_block.pregain_int)
        self.set_control_field_cb("dry_gain", lambda: self.dsp_block.dry_int)
        self.set_control_field_cb("predelay", lambda: self.dsp_block._predelay._delay)

        self.set_constant("sampling_freq", self.fs, "int32_t")
        self.set_constant("max_room_size", float(max_room_size), "float")
        self.set_constant("max_predelay", self.dsp_block._predelay._max_delay , "uint32_t")

        self.stage_memory_parameters = (
            self.constants["sampling_freq"],
            self.constants["max_room_size"],
            self.constants["max_predelay"]
        )

    def set_predelay(self, predelay):
        """
        Set the predelay of the wet channel.

        Parameters
        ----------
        predelay : float
            Predelay in ms, less than max_predelay.
        """
        self.dsp_block.predelay = predelay

    def set_wet_gain(self, gain_dB):
        """
        Set the wet gain of the reverb room stage. This sets the level
        of the reverberated signal.

        Parameters
        ----------
        gain_db : float
            Wet gain in dB, less than 0 dB.
        """
        self.dsp_block.wet_db = gain_dB

    def set_dry_gain(self, gain_dB):
        """
        Set the dry gain of the reverb room stage. This sets the level
        of the unprocessed signal.

        Parameters
        ----------
        gain_db : float
            Dry gain in dB, less than 0 dB.
        """
        self.dsp_block.dry_db = gain_dB

    def set_pre_gain(self, pre_gain):
        """
        Set the pre gain of the reverb room stage. It is not advised to
        increase this value above the default 0.015, as it can result in
        saturation inside the reverb delay lines.

        Parameters
        ----------
        pre_gain : float
            Pre gain value. Must be less than 1 (default 0.015).
        """
        self.dsp_block.pregain = pre_gain

    def set_room_size(self, new_room_size):
        """
        Set the room size, will adjust the delay line lengths.

        The room size is proportional to ``max_room_size``, and must be
        between 0 and 1. To increase the room_size above 1.0,
        ``max_room_size`` must instead be increased. Optimal memory
        usage occurs when ``room_size`` is set to 1.0.

        Parameters
        ----------
        new_room_size : float
            How big the room is as a proportion of max_room_size. This
            sets delay line lengths and must be between 0 and 1.
        """
        self.dsp_block.room_size = new_room_size

    def set_damping(self, damping):
        """
        Set the damping of the reverb room stage. This controls how much
        high frequency attenuation is in the room. Higher values yield
        shorter reverberation times at high frequencies.

        Parameters
        ----------
        damping : float
            How much high frequency attenuation in the room, between 0 and 1.
        """
        self.dsp_block.damping = damping

    def set_decay(self, decay):
        """
        Set the decay of the reverb room stage. This sets how
        reverberant the room is. Higher values will give a longer
        reverberation time for a given room size.

        Parameters
        ----------
        decay : float
            How long the reverberation of the room is, between 0 and 1.
        """
        self.dsp_block.decay = decay
