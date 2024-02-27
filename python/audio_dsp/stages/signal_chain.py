# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from ..design.stage import Stage, find_config, StageOutput
from operator import itemgetter
import audio_dsp.dsp.signal_chain as sc

class Bypass(Stage):
    """
    Stage which does not modify its inputs. Useful if data needs to flow through
    a thread which is not being processed on to keep pipeline lengths aligned.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("bypass"), **kwargs)
        self.create_outputs(self.n_in)


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
        super().__init__(config=find_config("fork"), **kwargs)
        self.create_outputs(self.n_in * count)

        fork_indices = [list(range(i, self.n_in * count, count)) for i in range(count)]
        self.forks = []
        for indices in fork_indices:
            self.forks.append([self.o[i] for i in indices])

    def get_frequency_response(self, nfft=512):
        # not sure what this looks like!
        raise NotImplementedError


class Mixer(Stage):
    """
    Mixes the input signals together. The mixer can be used to add signals
    together, or to attenuate the input signals.

    Attributes
    ----------
    gain_db : float
        The gain of the mixer in dB.
    """

    def __init__(self, gain_db=-6, **kwargs):
        super().__init__(config=find_config("mixer"), **kwargs)
        self.fs = int(self.fs)
        self.gain_db = gain_db
        self.create_outputs(1)
        self.dsp_block = sc.mixer(self.fs, self.n_in, gain_db)
        self.set_control_field_cb(
            "gain", lambda: [i for i in self.get_fixed_point_coeffs()]
        )


class Adder(Stage):
    """
    Add the input signals together. The adder can be used to add signals
    together, or to attenuate the input signals.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("adder"), **kwargs)
        self.fs = int(self.fs)
        self.create_outputs(1)
        self.dsp_block = sc.adder(self.fs, self.n_in)


class Subtractor(Stage):
    """
    Subtract the second input from the first. The subtractor can be used to
    subtract signals from each other.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("subtractor"), **kwargs)
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.dsp_block = sc.subtractor(self.fs, self.n_in)


class FixedGain(Stage):
    """
    Multiply the input by a fixed gain. The gain is set at the time of
    construction and cannot be changed.

    """

    def __init__(self, gain_db=-6, **kwargs):
        super().__init__(config=find_config("fixed_gain"), **kwargs)
        self.fs = int(self.fs)
        self.gain_db = gain_db
        self.create_outputs(self.n_in)
        self.dsp_block = sc.fixed_gain(self.fs, self.n_in, gain_db)


class VolumeControl(Stage):
    """
    Multiply the input by a gain. The gain can be changed at runtime.

    """

    def __init__(self, gain_db=-6, **kwargs):
        super().__init__(config=find_config("volume_control"), **kwargs)
        self.fs = int(self.fs)
        self.gain_db = gain_db
        self.create_outputs(self.n_in)
        self.dsp_block = sc.volume_control(self.fs, self.n_in, gain_db)
        self.set_control_field_cb(
            "gain", lambda: self.dsp_block.set_gain
        )

class Switch(Stage):
    """
    Switch the input to one of the outputs. The switch can be used to
    select between different signals.

    """

    def __init__(self, index=0, **kwargs):
        super().__init__(config=find_config("switch"), **kwargs)
        self.fs = int(self.fs)
        self.index = index
        self.create_outputs(self.n_in)
        self.dsp_block = sc.switch(self.fs, self.n_in)
        self.set_control_field_cb(
            "position", lambda: self.dsp_block.move_switch
        )