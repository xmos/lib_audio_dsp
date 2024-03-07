# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from ..design.stage import Stage, find_config, StageOutput
from operator import itemgetter


class Bypass(Stage):
    """
    Stage which does not modify its inputs. Useful if data needs to flow through
    a thread which is not being processed on to keep pipeline lengths aligned.
    """

    def __init__(self, **kwargs):
        super().__init__(name="bypass", **kwargs)
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
        self.forks: list[list[StageOutput]] = [list(itemgetter(*i)(self.o)) for i in fork_indices]

    def get_frequency_response(self, nfft=512):
        # not sure what this looks like!
        raise NotImplementedError
