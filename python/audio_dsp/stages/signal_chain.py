# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from ..design.stage import Stage, find_config
from ..dsp import generic as dspg
import audio_dsp.dsp.signal_chain as sc
import numpy as np


class Bypass(Stage):
    """
    Stage which does not modify its inputs. Useful if data needs to flow through
    a thread which is not being processed on to keep pipeline lengths aligned.
    """

    def __init__(self, **kwargs):
        super().__init__(name="bypass", **kwargs)
        self.create_outputs(self.n_in)

    def process(self, in_channels):
        return [np.copy(i) for i in in_channels]


class Fork(Stage):
    """
    Fork the signal, use if the same data needs to go down parallel
    data paths::

        a = t.stage(Example, ...)
        f = t.stage(Fork, a.o, count=2)  # count optional, default is 2
        b = t.stage(Example, f.forks[0])
        c = t.stage(Example, f.forks[1])

    Attributes
    ----------
    forks : list[list[StageOutput]]
        For convenience, each forked output will be available in this list
        each entry contains a set of outputs which will contain the same
        data as the input.
    """

    def __init__(self, count=2, **kwargs):
        super().__init__(name="fork", **kwargs)
        self.create_outputs(self.n_in * count)

        fork_indices = [list(range(i, self.n_in * count, count)) for i in range(count)]
        self.forks = []
        for indices in fork_indices:
            self.forks.append([self.o[i] for i in indices])

    def get_frequency_response(self, nfft=512):
        # not sure what this looks like!
        raise NotImplementedError

    def process(self, in_channels):
        n_forks = self.n_out // self.n_in
        ret = []
        for input in in_channels:
            for _ in range(n_forks):
                ret.append(np.copy(input))

        return ret


class Mixer(Stage):
    """
    Mixes the input signals together. The mixer can be used to add signals
    together, or to attenuate the input signals.
    """

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


class Adder(Stage):
    """
    Add the input signals together. The adder can be used to add signals
    together, or to attenuate the input signals.

    """

    def __init__(self, **kwargs):
        super().__init__(name="adder", **kwargs)
        self.create_outputs(1)
        self.dsp_block = sc.adder(self.fs, self.n_in)


class Subtractor(Stage):
    """
    Subtract the second input from the first. The subtractor can be used to
    subtract signals from each other. It must only have 2 inputs.

    """

    def __init__(self, **kwargs):
        super().__init__(name="subtractor", **kwargs)
        self.create_outputs(1)
        if self.n_in != 2:
            raise ValueError(f"Subtractor requires 2 inputs, got {self.n_in}")
        self.dsp_block = sc.subtractor(self.fs)


class FixedGain(Stage):
    """
    Multiply the input by a fixed gain. The gain is set at the time of
    construction and cannot be changed.

    Parameters
    ----------
    gain_db : float, optional
        The gain of the mixer in dB.

    """

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


class VolumeControl(Stage):
    """
    Multiply the input by a gain. The gain can be changed at runtime.

    Parameters
    ----------
    gain_db : float, optional
        The gain of the mixer in dB.

    """

    def __init__(self, gain_dB=0, **kwargs):
        super().__init__(config=find_config("volume_control"), **kwargs)
        self.create_outputs(self.n_in)
        slew_shift = 7
        self.dsp_block = sc.volume_control(self.fs, self.n_in, gain_dB, slew_shift)
        self.set_control_field_cb("target_gain", lambda: self.dsp_block.target_gain_int)
        self.set_control_field_cb("slew_shift", lambda: self.dsp_block.slew_shift)
        self.set_control_field_cb("mute", lambda: np.int32(self.dsp_block.mute_state))

    def make_volume_control(self, gain_dB, slew_shift, Q_sig=dspg.Q_SIG):
        self.details = dict(target_gain=gain_dB, slew_shift=slew_shift, Q_sig=Q_sig)
        self.dsp_block = sc.volume_control(self.fs, self.n_in, gain_dB, slew_shift, Q_sig)
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

    def mute(self, mute_state):
        """
        Set the mute state of the volume control.

        Parameters
        ----------
        mute_state : float
            The mute state of the volume control.
        """
        if mute_state:
            self.dsp_block.mute()
        else:
            self.dsp_block.unmute()
        return self


class Switch(Stage):
    """
    Switch the input to one of the outputs. The switch can be used to
    select between different signals.

    """

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
            The position to move the switch to.
        """
        self.dsp_block.move_switch(position)
        return self
