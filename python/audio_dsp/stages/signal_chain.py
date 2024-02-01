
from ..design.stage import Stage, find_config, StageOutput
from itertools import chain

class Bypass(Stage):
    """
    Stage which does not modify its inputs. Useful if data needs to flow through
    a thread which it is not being processed on to keep pipeline lengths aligned.
    """
    def __init__(self, **kwargs):
        super().__init__(config=find_config("bypass"), **kwargs)
        self.create_outputs(self.n_in)


class Fork(Stage):
    """
    Fork the signal, use if the same data needs to go down parallel 
    data paths

        a = t.stage(Example, ...)
        f = t.stage(Fork, a.o, count=2)  # count optional, default is 2
        b = t.stage(Example, f.forks[0])
        c = t.stage(Example, f.forks[1])

    Attributes
    ----------
    forks : list[StageOutput]
        For convenience, each forked output will be available in this list
        each entry contains a set of outputs which will contain the same
        data as the input.
    """
    def __init__(self, count=2, **kwargs):
        super().__init__(config=find_config("fork"), **kwargs)
        self.create_outputs(self.n_in * count)
        self.forks = [self.o[i:i + self.n_in] for i in range(0, self.n_in * count, self.n_in)]
