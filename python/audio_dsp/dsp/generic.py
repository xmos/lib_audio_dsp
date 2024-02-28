# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from copy import deepcopy

import numpy as np
from audio_dsp.dsp import utils as utils
from docstring_inheritance import NumpyDocstringInheritanceInitMeta

Q_SIG = 27
HEADROOM_BITS = 31 - Q_SIG
HEADROOM_DB = utils.db(2**HEADROOM_BITS)


class dsp_block(metaclass=NumpyDocstringInheritanceInitMeta):
    """
    Generic DSP block, all blocks should inherit from this class and
    implement it's methods.

    By using the metaclass NumpyDocstringInheritanceInitMeta, parameter
    and attribute documentation can be inherited by the child classes.

    Parameters
    ----------
    fs : int
        sampling frequency in Hz.
    n_chans : int
        number of channels the block runs on.
    Q_sig: int, optional
        Q format of the signal, number of bits after the decimal point.
        Defaults to Q27.

    Attributes
    ----------
    fs : int
        sampling frequency in Hz.
    n_chans : int
        number of channels the block runs on.
    Q_sig: int
        Q format of the signal, number of bits after the decimal point.
    """

    def __init__(self, fs, n_chans, Q_sig=Q_SIG):
        self.fs = fs
        self.n_chans = n_chans
        self.Q_sig = Q_sig
        return

    def process(self, sample: float, channel=0):
        """
        Take one new sample and give it back. Do no processing for the
        generic block.
        """
        return sample

    def process_xcore(self, sample: float, channel=0):
        """
        Take one new sample and return 1 processed sample.

        For the generic implementation, scale and quantize the input,
        call the float implementation, then scale back to 1.0 = 0 dB.
        """
        sample_int = utils.int32(sample * 2**self.Q_sig)
        y = self.process(float(sample_int))
        y_flt = float(y) * 2**-self.Q_sig

        return y_flt

    def process_frame(self, frame: list):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        For the generic implementation, just call process for each
        sample for each channel.
        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process(this_chan[sample], channel=chan)

        return output

    def process_frame_xcore(self, frame: list):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        For the generic implementation, just call process for each
        sample for each channel.
        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_xcore(this_chan[sample], channel=chan)

        return output

    def freq_response(self, nfft=512):
        """
        The frequency response of the module for a nominal input.

        The generic module has a flat frequency response.
        """
        f = np.fft.rfftfreq(nfft) * self.fs
        h = np.ones_like(f)
        return f, h
