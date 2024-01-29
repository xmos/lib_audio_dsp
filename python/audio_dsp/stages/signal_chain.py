
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


class Sum(Stage):
    """
    Adds channels together. 

    Below are two examples, "a" and "b" represent stages which have the same
    number of output channels, lets say 2. The first examples sums the stereo
    inputs to make a stereo output. The second sums all four channels to make
    a mono output.

        t.stage(Sum, [a.o, b.o])
        t.stage(Sum, [*a.o, *b.o])

    The decision is based on whether inputs is a list of StageOutputs, or a list
    containing lists of StageOutputs.
    """
    def __init__(self, inputs, **kwargs):
        if not isinstance(inputs[0], StageOutput):
            flat_inputs = list(chain.from_iterable(zip(*inputs)))
            n_out = len(inputs[0])
            # check all inputs have the same length
            lengths = [len(i) for i in inputs]
            for la, lb in zip(lengths[1:], lengths[:-1]):
                if la != lb:
                    raise ValueError(f"All inputs to sum must be the same length, got {', '.join(str(i) for i in lengths)}")
        else:
            flat_inputs = inputs
            n_out = 1
        super().__init__(inputs = flat_inputs, config=find_config("sum"), **kwargs)
        self.create_outputs(n_out)

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