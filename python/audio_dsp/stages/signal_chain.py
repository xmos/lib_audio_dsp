# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Signal chain stages allow for the control of signal flow through the
pipeline. This includes stages for combining and splitting signals, basic
gain components, and delays.
"""

from ..design.stage import (
    Stage,
    find_config,
    StageOutputList,
    StageOutput,
    StageParameters,
    StageConfig,
)

from ..dsp import generic as dspg
import audio_dsp.dsp.signal_chain as sc
import numpy as np

from typing import Literal
from pydantic import Field

from audio_dsp.models.signal_chain import VolumeControlParameters, FixedGainParameters, SwitchParameters, MixerParameters

class Bypass(Stage):
    """
    Stage which does not modify its inputs. Useful if data needs to flow through
    a thread which is not being processed on to keep pipeline lengths aligned.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["Bypass"] = "Bypass"

    def __init__(self, **kwargs):
        super().__init__(name="bypass", **kwargs)
        self.create_outputs(self.n_in)

    def process(self, in_channels):
        """Return a copy of the inputs."""
        return [np.copy(i) for i in in_channels]


class ForkConfig(StageConfig):
    count: int = Field(default=1)


class Fork(Stage):
    """
    Fork the signal.

    Use if the same data needs to be sent to multiple data paths::

        a = t.stage(Example, ...)
        f = t.stage(Fork, a, count=2)  # count optional, default is 2
        b = t.stage(Example, f.forks[0])
        c = t.stage(Example, f.forks[1])

    Attributes
    ----------
    forks : list[list[StageOutput]]
        For convenience, each forked output will be available in this list
        each entry contains a set of outputs which will contain the same
        data as the input.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["Fork"] = "Fork"
    #     config: ForkConfig = Field(default_factory=ForkConfig)

    class ForkOutputList(StageOutputList):
        """
        Custom StageOutputList that is created by Fork.

        This allows convenient access to each fork output.

        Attributes
        ----------
        forks: list[StageOutputList]
            Fork duplicates its inputs, each entry in the forks list is a single copy
            of the input edges.
        """

        def __init__(self, edges: list[StageOutput | None] | None = None):
            super().__init__(edges)
            self.forks = []

    def __init__(self, count=2, **kwargs):
        super().__init__(name="fork", **kwargs)
        self.create_outputs(self.n_in * count)

        fork_indices = [list(range(i, self.n_in * count, count)) for i in range(count)]
        forks = []
        for indices in fork_indices:
            forks.append(self.o[(i for i in indices)])
        self._o = self.ForkOutputList(self.o.edges)
        self._o.forks = forks

    def get_frequency_response(self, nfft=512):
        """Fork has no sensible frequency response, not implemented."""
        # not sure what this looks like!
        raise NotImplementedError

    def process(self, in_channels):
        """Duplicate the inputs to the outputs based on this fork's configuration."""
        n_forks = self.n_out // self.n_in
        ret = []
        for input in in_channels:
            for _ in range(n_forks):
                ret.append(np.copy(input))

        return ret


class MixerParameters(StageParameters):
    gain_db: float = Field(default=0)


class Mixer(Stage):
    """
    Mixes the input signals together. The mixer can be used to add signals
    together, or to attenuate the input signals.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.mixer`
        The DSP block class; see :ref:`Mixer` for implementation details
    """

    # class Model(Stage.Model):
    #     op_type: Literal["Mixer"] = "Mixer"
    #     parameters: MixerParameters = Field(default_factory=MixerParameters)

    def __init__(self, **kwargs):
        super().__init__(config=find_config("mixer"), **kwargs)
        self.create_outputs(1)
        self.dsp_block = sc.mixer(self.fs, self.n_in)
        self.set_control_field_cb("gain", lambda: self.dsp_block.gain_int)

    def set_gain(self, gain_db):
        """
        Set the gain of the mixer in dB.

        Parameters
        ----------
        gain_db : float
            The gain of the mixer in dB.
        """
        self.dsp_block = sc.mixer(self.fs, self.n_in, gain_db)
        return self

    def set_parameters(self, parameters: MixerParameters):
        self.set_gain(parameters.gain_db)

class Adder(Stage):
    """
    Add the input signals together. The adder can be used to add signals
    together.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.adder`
        The DSP block class; see :ref:`Adder` for implementation details.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["Adder"] = "Adder"

    def __init__(self, **kwargs):
        super().__init__(name="adder", **kwargs)
        self.create_outputs(1)
        self.dsp_block = sc.adder(self.fs, self.n_in)


class Subtractor(Stage):
    """
    Subtract the second input from the first. The subtractor can be used to
    subtract signals from each other. It must only have 2 inputs.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.subtractor`
        The DSP block class; see :ref:`Subtractor` for implementation details.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["Subtractor"] = "Subtractor"

    def __init__(self, **kwargs):
        super().__init__(name="subtractor", **kwargs)
        self.create_outputs(1)
        if self.n_in != 2:
            raise ValueError(f"Subtractor requires 2 inputs, got {self.n_in}")
        self.dsp_block = sc.subtractor(self.fs)


class FixedGainParameters(StageParameters):
    gain_db: float = Field(default=0)


class FixedGain(Stage):
    """
    This stage implements a fixed gain. The input signal is multiplied
    by a gain. If the gain is changed at runtime, pops and clicks may
    occur.

    If the gain needs to be changed at runtime, use a
    :class:`VolumeControl` stage instead.

    Parameters
    ----------
    gain_db : float, optional
        The gain of the mixer in dB.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.fixed_gain`
        The DSP block class; see :ref:`FixedGain` for implementation details.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["FixedGain"] = "FixedGain"
    #     parameters: FixedGainParameters = Field(default_factory=FixedGainParameters)

    def __init__(self, gain_db=0, **kwargs):
        super().__init__(config=find_config("fixed_gain"), **kwargs)
        self.create_outputs(self.n_in)
        self.dsp_block = sc.fixed_gain(self.fs, self.n_in, gain_db)
        self.set_control_field_cb("gain", lambda: self.dsp_block.gain_int)

    def set_gain(self, gain_db):
        """
        Set the gain of the fixed gain in dB.

        Parameters
        ----------
        gain_db : float
            The gain of the fixed gain in dB.
        """
        self.dsp_block = sc.fixed_gain(self.fs, self.n_in, gain_db)
        return self

    def set_parameters(self, parameters: FixedGainParameters):
        self.set_gain(parameters.gain_db)


class VolumeControl(Stage):
    """
    This stage implements a volume control. The input signal is
    multiplied by a gain. The gain can be changed at runtime. To avoid
    pops and clicks during gain changes, a slew is applied to the gain
    update. The stage can be muted and unmuted at runtime.

    Parameters
    ----------
    gain_db : float, optional
        The gain of the mixer in dB.
    mute_state : int, optional
        The mute state of the Volume Control: 0: unmuted, 1: muted.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.volume_control`
        The DSP block class; see :ref:`VolumeControl` for implementation details.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["VolumeControl"] = "VolumeControl"

    def __init__(self, gain_dB=0, mute_state=0, **kwargs):
        super().__init__(config=find_config("volume_control"), **kwargs)
        self.create_outputs(self.n_in)
        slew_shift = 7
        self.dsp_block = sc.volume_control(self.fs, self.n_in, gain_dB, slew_shift, mute_state)
        self.set_control_field_cb("target_gain", lambda: self.dsp_block.target_gain_int)
        self.set_control_field_cb("slew_shift", lambda: self.dsp_block.slew_shift)
        self.set_control_field_cb("mute_state", lambda: np.int32(self.dsp_block.mute_state))

        self.stage_memory_string = "volume_control"
        self.stage_memory_parameters = (self.n_in,)

    def make_volume_control(self, gain_dB, slew_shift, mute_state, Q_sig=dspg.Q_SIG):
        """
        Update the settings of this volume control.

        Parameters
        ----------
        gain_dB
            Target gain of this volume control.
        slew_shift
            The shift value used in the exponential slew.
        mute_state
            The mute state of the Volume Control: 0: unmuted, 1: muted.
        """
        self.details = dict(
            target_gain=gain_dB, slew_shift=slew_shift, mute_state=mute_state, Q_sig=Q_sig
        )
        self.dsp_block = sc.volume_control(
            self.fs, self.n_in, gain_dB, slew_shift, mute_state, Q_sig
        )
        return self

    def set_gain(self, gain_dB):
        """
        Set the gain of the volume control in dB.

        Parameters
        ----------
        gain_db : float
            The gain of the volume control in dB.
        """
        self.dsp_block.set_gain(gain_dB)
        return self

    def set_mute_state(self, mute_state):
        """
        Set the mute state of the volume control.

        Parameters
        ----------
        mute_state : bool
            The mute state of the volume control.
        """
        if mute_state:
            self.dsp_block.mute()
        else:
            self.dsp_block.unmute()
        return self

    def set_parameters(self, parameters: VolumeControlParameters):
        self.set_gain(parameters.gain_db)
        self.set_mute_state(parameters.mute_state)

class Switch(Stage):
    """
    Switch the input to one of the outputs. The switch can be used to
    select between different signals.

    """

    # class Model(Stage.Model):
    #     op_type: Literal["Switch"] = "Switch"

    def __init__(self, index=0, **kwargs):
        super().__init__(config=find_config("switch"), **kwargs)
        self.index = index
        self.create_outputs(1)
        self.dsp_block = sc.switch(self.fs, self.n_in)
        self.set_control_field_cb("position", lambda: self.dsp_block.switch_position)

    def move_switch(self, position):
        """
        Move the switch to the specified position.

        Parameters
        ----------
        position : int
            The position to which to move the switch. This changes the output
            signal to the input[position]
        """
        self.dsp_block.move_switch(position)
        return self

    def set_parameters(self, parameters: SwitchParameters):
        self.move_switch(parameters.position)

class SwitchStereo(Stage):
    """
    Switch the input to one of the stereo pairs of outputs. The switch
    can be used to select between different stereo signal pairs. The
    inputs should be passed in pairs, e.g. ``[0_L, 0_R, 1_L, 1_R, ...]``.
    Setting the switch position will output the nth pair.

    """

    # class Model(Stage.Model):
    #     op_type: Literal["SwitchStereo"] = "SwitchStereo"

    def __init__(self, index=0, **kwargs):
        super().__init__(config=find_config("switch_stereo"), **kwargs)
        self.index = index
        self.create_outputs(2)
        self.dsp_block = sc.switch_stereo(self.fs, self.n_in)
        self.set_control_field_cb("position", lambda: self.dsp_block.switch_position)

    def move_switch(self, position):
        """
        Move the switch to the specified position.

        Parameters
        ----------
        position : int
            The position to which to move the switch. This changes the output
            signal to the [input[2*position], input[:2*position + 1]]
        """
        self.dsp_block.move_switch(position)
        return self

    def set_parameters(self, parameters: SwitchParameters):
        self.move_switch(parameters.position)

class Delay(Stage):
    """
    Delay the input signal by a specified amount.

    The maximum delay is set at compile time, and the runtime delay can
    be set between 0 and ``max_delay``.

    Parameters
    ----------
    max_delay : float
        The maximum delay in specified units. This can only be set at
        compile time.
    starting_delay : float
        The starting delay in specified units.
    units : str, optional
        The units of the delay, can be 'samples', 'ms' or 's'.
        Default is 'samples'.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.signal_chain.delay`
        The DSP block class; see :ref:`Delay` for implementation details.
    """

    # class Model(Stage.Model):
    #     op_type: Literal["Delay"] = "Delay"

    def __init__(self, max_delay, starting_delay, units="samples", **kwargs):
        super().__init__(config=find_config("delay"), **kwargs)
        self.create_outputs(self.n_in)
        self.dsp_block = sc.delay(self.fs, self.n_in, max_delay, starting_delay, units)
        self["max_delay"] = max_delay
        self.set_control_field_cb("max_delay", lambda: self.dsp_block._max_delay)
        self.set_control_field_cb("delay", lambda: self.dsp_block.delay)

        self.stage_memory_parameters = (self.n_in, self["max_delay"])

    def set_delay(self, delay, units="samples"):
        """
        Set the length of the delay line, will saturate at max_delay.

        Parameters
        ----------
        delay : float
            The delay in specified units.
        units : str
            The units of the delay, can be 'samples', 'ms' or 's'.
            Default is 'samples'.
        """
        self.dsp_block.set_delay(delay, units)
