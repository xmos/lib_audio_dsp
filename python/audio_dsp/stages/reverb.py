# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Reverb Stages emulate the natural reverberance of rooms."""

from audio_dsp.design.stage import Stage, find_config
import audio_dsp.dsp.reverb as rvrb
import audio_dsp.dsp.reverb_stereo as rvbs
import audio_dsp.dsp.reverb_plate as rvp

from audio_dsp.models.reverb import (
    ReverbRoomParameters,
    ReverbRoomStereoParameters,
    ReverbPlateParameters,
    ReverbRoomConfig,
    ReverbPlateConfig
)


class _ReverbBase(Stage):
    """
    The base class for reverb stages, containing pre delays, and wet/dry
    mixes and pregain.
    """

    def set_wet_dry_mix(self, mix):
        """
        Set the wet/dry gains so that the mix of 0 results in a
        fully dry output, the mix of 1 results in a fully wet output.

        Parameters
        ----------
        mix : float
            The wet/dry mix, must be [0, 1].
        """
        parameters = self.parameters.copy(update={"wet_dry_mix": mix})
        self.set_parameters(parameters)

    def set_predelay(self, predelay):
        """
        Set the predelay of the wet channel.

        Parameters
        ----------
        predelay : float
            Predelay in ms, less than max_predelay.
        """
        parameters = self.parameters.copy(update={"predelay": predelay})
        self.set_parameters(parameters)

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
        Set the pre gain of the reverb room stage.

        Parameters
        ----------
        pre_gain : float
            Pre gain value. Must be less than 1.
        """
        parameters = self.parameters.copy(update={"pregain": pre_gain})
        self.set_parameters(parameters)


class ReverbRoom(_ReverbBase):
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
        if self.n_in != 1:
            raise ValueError("ReverbRoom must have 1 channel")
        self.create_outputs(self.n_in)

        max_predelay = predelay if max_predelay == None else max_predelay

        self.config = ReverbRoomConfig(max_room_size=max_room_size, max_predelay=max_predelay)

        self.parameters = ReverbRoomParameters(
            room_size=1.0,
            damping=0.4,
            decay=0.5,
            predelay=predelay,
            pregain=0.015,
            wet_dry_mix=0.5,
        )

        self.dsp_block: rvrb.reverb_room = rvrb.reverb_room(
            self.fs,
            self.n_in,
            max_room_size=max_room_size,
            predelay=predelay,
            max_predelay=max_predelay,
        )
        self.set_control_field_cb("room_size", lambda: self.dsp_block.room_size)
        self.set_control_field_cb("feedback", lambda: self.dsp_block.combs[0].feedback_int)
        self.set_control_field_cb("damping", lambda: self.dsp_block.combs[0].damp1_int)
        self.set_control_field_cb("wet_gain", lambda: self.dsp_block.wet_int)
        self.set_control_field_cb("pregain", lambda: self.dsp_block.pregain_int)
        self.set_control_field_cb("dry_gain", lambda: self.dsp_block.dry_int)
        self.set_control_field_cb("predelay", lambda: self.dsp_block._predelay._delay)

        self.set_constant("sampling_freq", self.fs, "int32_t")
        self.set_constant("max_room_size", float(max_room_size), "float")
        self.set_constant("max_predelay", self.dsp_block._predelay._max_delay, "uint32_t")

        self.stage_memory_parameters = (
            self.constants["sampling_freq"],
            self.constants["max_room_size"],
            self.constants["max_predelay"],
        )

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
        parameters = self.parameters.copy(update={"room_size": new_room_size})
        self.set_parameters(parameters)

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
        parameters = self.parameters.copy(update={"damping": damping})
        self.set_parameters(parameters)

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
        parameters = self.parameters.copy(update={"decay": decay})
        self.set_parameters(parameters)

    def set_parameters(self, parameters: ReverbRoomParameters):
        """Update the parameters of the ReverbRoom stage."""
        self.parameters = parameters
        self.dsp_block.room_size = parameters.room_size
        self.dsp_block.damping = parameters.damping
        self.dsp_block.decay = parameters.decay
        self.dsp_block.predelay = parameters.predelay
        self.dsp_block.pregain = parameters.pregain
        self.dsp_block.set_wet_dry_mix(parameters.wet_dry_mix)


class ReverbRoomStereo(ReverbRoom):
    """
    The stereo room reverb stage. This is based on Freeverb by Jezar at
    Dreampoint. Each channel consists of 8 parallel comb filters fed
    into 4 series all-pass filters, and the reverberator outputs are mixed
    according to the ``width`` parameter.

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
    dsp_block : :class:`audio_dsp.dsp.reverb_stereo.reverb_room_stereo`
        The DSP block class; see :ref:`ReverbRoomStereo`
        for implementation details.
    """

    def __init__(self, max_room_size=1, predelay=10, max_predelay=None, **kwargs):
        Stage.__init__(self, config=find_config("reverb_room_stereo"), **kwargs)
        if self.fs is None:
            raise ValueError("Reverb requires inputs with a valid fs")
        if self.n_in != 2:
            raise ValueError("ReverbRoomStereo must have 2 channels")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)

        max_predelay = predelay if max_predelay == None else max_predelay

        self.config = ReverbRoomStereoConfig(max_room_size=max_room_size, max_predelay=max_predelay)

        self.parameters = ReverbRoomStereoParameters(
            room_size=1.0,
            damping=0.4,
            decay=0.5,
            predelay=predelay,
            pregain=0.015,
            wet_dry_mix=0.5,
            width=0.5,
        )

        self.dsp_block: rvbs.reverb_room_stereo = rvbs.reverb_room_stereo(
            self.fs,
            self.n_in,
            max_room_size=max_room_size,
            predelay=predelay,
            max_predelay=max_predelay,
        )
        self.set_control_field_cb("room_size", lambda: self.dsp_block.room_size)
        self.set_control_field_cb("feedback", lambda: self.dsp_block.combs_l[0].feedback_int)
        self.set_control_field_cb("damping", lambda: self.dsp_block.combs_l[0].damp1_int)
        self.set_control_field_cb("wet_gain1", lambda: self.dsp_block.wet_1_int)
        self.set_control_field_cb("wet_gain2", lambda: self.dsp_block.wet_2_int)
        self.set_control_field_cb("pregain", lambda: self.dsp_block.pregain_int)
        self.set_control_field_cb("dry_gain", lambda: self.dsp_block.dry_int)
        self.set_control_field_cb("predelay", lambda: self.dsp_block._predelay._delay)

        self.set_constant("sampling_freq", self.fs, "int32_t")
        self.set_constant("max_room_size", float(max_room_size), "float")
        self.set_constant("max_predelay", self.dsp_block._predelay._max_delay, "uint32_t")

        self.stage_memory_parameters = (
            self.constants["sampling_freq"],
            self.constants["max_room_size"],
            self.constants["max_predelay"],
        )

    def set_width(self, width):
        """
        Set the decay of the reverb room stage. This sets how
        reverberant the room is. Higher values will give a longer
        reverberation time for a given room size.

        Parameters
        ----------
        width : float
            How much stereo separation between the channels. A width of 0
            indicates no stereo separation (i.e. mono). A width of 1 indicates
            maximum stereo separation.
        """
        parameters = self.parameters.copy(update={"width": width})
        self.set_parameters(parameters)

    def set_parameters(self, parameters: ReverbRoomStereoParameters):  # pyright: ignore overload
        """Update the parameters of the ReverbRoomStereo stage."""
        self.parameters = parameters
        self.dsp_block.room_size = parameters.room_size
        self.dsp_block.damping = parameters.damping
        self.dsp_block.decay = parameters.decay
        self.dsp_block.predelay = parameters.predelay
        self.dsp_block.pregain = parameters.pregain
        self.dsp_block.set_wet_dry_mix(parameters.wet_dry_mix)
        self.dsp_block.width = parameters.width


class ReverbPlateStereo(_ReverbBase):
    """
    The stereo room plate stage. This is based on Dattorro's 1997
    paper. This reverb consists of 4 allpass filters for input diffusion,
    followed by a figure of 8 reverb tank of allpasses, low-pass filters,
    and delays. The output is taken from multiple taps in the delay lines
    to get a desirable echo density.

    Parameters
    ----------
    predelay : float, optional
        The delay applied to the wet channel in ms.
    max_predelay : float, optional
        The maximum predelay in ms.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.reverb.reverb_plate_stereo`
        The DSP block class; see :ref:`ReverbPlateStereo`
        for implementation details.
    """

    def __init__(self, predelay=10, max_predelay=None, **kwargs):
        super().__init__(config=find_config("reverb_plate_stereo"), **kwargs)
        if self.fs is None:
            raise ValueError("Reverb requires inputs with a valid fs")
        self.fs = int(self.fs)
        if self.n_in != 2:
            raise ValueError("ReverbPlateStereo must have 2 channels")
        self.create_outputs(self.n_in)

        max_predelay = predelay if max_predelay == None else max_predelay

        self.config = ReverbPlateConfig( max_predelay=max_predelay)

        self.parameters = ReverbPlateParameters(
            decay=0.4,
            damping=0.75,
            bandwidth=8000,
            early_diffusion=0.75,
            late_diffusion=0.7,
            width=1.0,
            wet_dry_mix=0.5,
            pregain=0.5,
            predelay=predelay,
        )

        self.dsp_block: rvp.reverb_plate_stereo = rvp.reverb_plate_stereo(
            self.fs,
            self.n_in,
            predelay=predelay,
            max_predelay=max_predelay,
        )
        self.set_control_field_cb("damping", lambda: self.dsp_block.lowpasses[1].coeff_b0_int)
        self.set_control_field_cb(
            "late_diffusion", lambda: self.dsp_block.mod_allpasses[0].feedback_int
        )
        self.set_control_field_cb("bandwidth", lambda: self.dsp_block.lowpasses[0].coeff_b0_int)
        self.set_control_field_cb(
            "early_diffusion", lambda: self.dsp_block.allpasses[0].feedback_int
        )

        self.set_control_field_cb("decay", lambda: self.dsp_block.decay_int)
        self.set_control_field_cb("wet_gain1", lambda: self.dsp_block.wet_1_int)
        self.set_control_field_cb("wet_gain2", lambda: self.dsp_block.wet_2_int)
        self.set_control_field_cb("pregain", lambda: self.dsp_block.pregain_int)
        self.set_control_field_cb("dry_gain", lambda: self.dsp_block.dry_int)
        self.set_control_field_cb("predelay", lambda: self.dsp_block._predelay._delay)

        self.set_constant("sampling_freq", self.fs, "int32_t")
        self.set_constant("max_predelay", self.dsp_block._predelay._max_delay, "uint32_t")

        self.stage_memory_parameters = (
            self.constants["sampling_freq"],
            self.constants["max_predelay"],
        )

    def set_width(self, width):
        """
        Set the decay of the reverb room stage. This sets how
        reverberant the room is. Higher values will give a longer
        reverberation time for a given room size.

        Parameters
        ----------
        width : float
            How much stereo separation between the channels. A width of 0
            indicates no stereo separation (i.e. mono). A width of 1 indicates
            maximum stereo separation.
        """
        parameters = self.parameters.copy(update={"width": width})
        self.set_parameters(parameters)

    def set_damping(self, damping):
        """
        Set the damping of the plate reverb stage. This controls how much
        high frequency attenuation is in the plate. Higher values yield
        shorter reverberation times at high frequencies.

        Parameters
        ----------
        damping : float
            How much high frequency attenuation in the plate, between 0 and 1.
        """
        parameters = self.parameters.copy(update={"damping": damping})
        self.set_parameters(parameters)

    def set_decay(self, decay):
        """
        Set the decay of the plate reverb stage. This sets how
        reverberant the plate is. Higher values will give a longer
        reverberation time.

        Parameters
        ----------
        decay : float
            How long the reverberation of the plate is, between 0 and 1.
        """
        parameters = self.parameters.copy(update={"decay": decay})
        self.set_parameters(parameters)

    def set_early_diffusion(self, diffusion):
        """
        Set the early diffusion of the plate reverb stage. This sets how
        much diffusion is present in the first part of the reverberation.
        Higher values will give more diffusion.

        Parameters
        ----------
        diffusion : float
            How diffuse the plate is, between 0 and 1.
        """
        parameters = self.parameters.copy(update={"early_diffusion": diffusion})
        self.set_parameters(parameters)

    def set_late_diffusion(self, diffusion):
        """
        Set the late diffusion of the plate reverb stage. This sets how
        much diffusion is present in the latter part of the reverberation.
        Higher values will give more diffusion.

        Parameters
        ----------
        diffusion : float
            How diffuse the plate is, between 0 and 1.
        """
        parameters = self.parameters.copy(update={"late_diffusion": diffusion})
        self.set_parameters(parameters)

    def set_bandwidth(self, bandwidth):
        """
        Set the bandwidth of the plate reverb stage. This sets the low
        pass cutoff frequency of the reverb input. Higher values will
        give a higher cutoff frequency.

        Parameters
        ----------
        bandwidth : float
            The bandwidth of the plate input signal, between 0 and 1.
        """
        parameters = self.parameters.copy(update={"bandwidth": bandwidth})
        self.set_parameters(parameters)

    def set_parameters(self, parameters: ReverbPlateParameters):
        """Update the parameters of the ReverbPlate stage."""
        self.parameters = parameters
        self.dsp_block.damping = parameters.damping
        self.dsp_block.decay = parameters.decay
        self.dsp_block.early_diffusion = parameters.early_diffusion
        self.dsp_block.late_diffusion = parameters.late_diffusion
        self.dsp_block.bandwidth = parameters.bandwidth
        self.dsp_block.width = parameters.width
        self.dsp_block.pregain = parameters.pregain
        self.dsp_block.predelay = parameters.predelay
        self.dsp_block.set_wet_dry_mix(parameters.wet_dry_mix)
