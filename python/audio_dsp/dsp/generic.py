# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The generic DSP block and globals."""

from copy import deepcopy

import numpy as np
from audio_dsp.dsp import utils as utils
from docstring_inheritance import NumpyDocstringInheritanceInitMeta

# default Q format for the signal
Q_SIG = 27

# number of bits for the headroom, this will set the maximum gain that
# can be applied to the signal without overflowing.
HEADROOM_BITS = 31 - Q_SIG
HEADROOM_DB = utils.db((utils.Q_max(31) + 1) / utils.Q_max(Q_SIG))


class dsp_block(metaclass=NumpyDocstringInheritanceInitMeta):
    """
    Generic DSP block, all blocks should inherit from this class and
    implement it's methods.

    By using the metaclass NumpyDocstringInheritanceInitMeta, parameter
    and attribute documentation can be inherited by the child classes.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    n_chans : int
        Number of channels the block runs on.
    Q_sig: int, optional
        Q format of the signal, number of bits after the decimal point.
        Defaults to Q4.27.

    Attributes
    ----------
    fs : int
        Sampling frequency in Hz.
    n_chans : int
        Number of channels the block runs on.
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

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int, optional
            The channel index to process the sample on. Default is 0.

        Returns
        -------
        float
            The processed sample.
        """
        raise NotImplementedError

    def process_xcore(self, sample: float, channel=0):
        """Take one new sample and return 1 processed sample.

        For the generic implementation, scale and quantize the input,
        call the xcore-like implementation, then scale back to 1.0 = 0 dB.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int, optional
            The channel index to process the sample on. Default is 0.

        Returns
        -------
        float
            The processed output sample.
        """
        sample_int = utils.float_to_fixed(sample, self.Q_sig)
        y = self.process(float(sample_int))
        y_flt = utils.fixed_to_float(y, self.Q_sig)

        return y_flt

    def process_channels(self, sample_list: list[float]) -> list[float]:
        """
        Process the sample in each audio channel using floating point maths.

        The generic implementation calls self.process for each channel.

        Parameters
        ----------
        sample_list : list[float]
            The input samples to be processed. Each sample represents a
            different channel

        Returns
        -------
        list[float]
            The processed samples for each channel.
        """
        output_samples = deepcopy(sample_list)
        for channel in range(len(output_samples)):
            output_samples[channel] = self.process(sample_list[channel], channel)
        return output_samples

    def process_channels_xcore(self, sample_list: list[float]) -> list[float]:
        """
        Process the sample in each audio channel using fixed point maths.

        The generic implementation calls self.process_xcore for each channel.

        Parameters
        ----------
        sample_list : list[float]
            The input samples to be processed. Each sample represents a
            different channel

        Returns
        -------
        list[float]
            The processed samples for each channel.
        """
        output_samples = deepcopy(sample_list)
        for channel in range(len(output_samples)):
            output_samples[channel] = self.process_xcore(sample_list[channel], channel)
        return output_samples

    def process_frame(self, frame: list):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        For the generic implementation, just call process for each
        sample for each channel.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.

        Returns
        -------
        list
            List of processed frames, with the same structure as the
            input frame.
        """
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = np.zeros((len(frame), frame_size))
        for sample in range(frame_size):
            output[:, sample] = self.process_channels(frame_np[:, sample].tolist())

        return list(output)

    def process_frame_xcore(self, frame: list):
        """
        Take a list frames of samples and return the processed frames,
        using an xcore-like implementation.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        For the generic implementation, just call process for each
        sample for each channel.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.

        Returns
        -------
        list
            List of processed frames, with the same structure as the
            input frame.
        """
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = np.zeros((len(frame), frame_size))
        for sample in range(frame_size):
            output[:, sample] = self.process_channels_xcore(frame_np[:, sample].tolist())

        return list(output)

    def freq_response(self, nfft=512):
        """
        Calculate the frequency response of the module for a nominal
        input.

        The generic module has a flat frequency response.

        Parameters
        ----------
        nfft : int, optional
            The number of points to use for the FFT, by default 512

        Returns
        -------
        tuple
            A tuple containing the frequency values and the
            corresponding complex response.

        """
        f = np.fft.rfftfreq(nfft) * self.fs
        h = np.ones_like(f)
        return f, h
