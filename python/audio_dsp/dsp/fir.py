# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""The FIR dsp block."""

import numpy as np
import warnings
from pathlib import Path
import sys
import os
import types

from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils


class fir_direct(dspg.dsp_block):
    """
    An FIR filter, implemented in direct form in the time domain.

    When the filter coefficients are converted to fixed point, if there
    will be leading zeros, a left shift is applied to the coefficients
    in order to use the full dynamic range of the VPU. A subsequent
    right shift is applied to the accumulator after the convolution to
    return to the same gain.

    Parameters
    ----------
    coeffs_path : Path
        Path to a file containing the coefficients, in a format
        supported by `np.loadtxt <https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_.

    Attributes
    ----------
    coeffs : np.ndarray
        Array of the FIR coefficients in floating point format.
    coeffs_int : list
        Array of the FIR coefficients in fixed point int32 format.
    shift : int
        Right shift to be applied to the fixed point convolution result.
        This compensates for any left shift applied to the coefficients.
    n_taps : int
        Number of taps in the filter.
    buffer : np.ndarray
        Buffer of previous inputs for the convlution in floating point
        format.
    buffer_int : list
        Buffer of previous inputs for the convlution in fixed point
        format.
    buffer_idx : list
        List of the floating point buffer head for each channel.
    buffer_idx_int : list
        List of the fixed point point buffer head for each channel.

    """

    def __init__(self, fs: float, n_chans: int, coeffs_path: Path, Q_sig: int = dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.coeffs = np.loadtxt(coeffs_path)
        self.n_taps = len(self.coeffs)
        self.coeffs_int, self.shift = self.check_coeff_scaling()

        self.reset_state()
        self.buffer_idx = [self.n_taps - 1] * self.n_chans
        self.buffer_idx_int = [self.n_taps - 1] * self.n_chans

    def check_coeff_scaling(self):
        """Check the coefficient scaling is optimal.

        If there will be leading zeros, calculate a shift to use the
        full dynamic range of the VPU
        """
        int32_max = 2**31 - 1

        # scale to Q30, to match VPU shift but keep as double for now
        # until we see how many bits we have
        scaled_coeffs = self.coeffs * (2**30)

        # find how many bits we can (or need to) shift the coeffs by
        max_coeff = np.max(np.abs(scaled_coeffs))
        coeff_headroom = max_coeff / int32_max
        coeff_headroom_bits = -np.ceil(np.log2(coeff_headroom))
        shift = coeff_headroom_bits

        # shift the scaled coeffs
        scaled_coeffs *= 2**coeff_headroom_bits

        # check the gain of the filter will fit in the output Q format
        headroom = utils.db(2 ** (31 - self.Q_sig))
        w, h = self.freq_response()
        coeff_max_gain = np.max(utils.db(h))

        if coeff_max_gain > headroom:
            warnings.warn(
                "Headroom of %d dB is not sufficient to guarentee no clipping." % (headroom)
            )

        # VPU stripes the convolution across 8 40b accumulators
        vpu_acc_max = 0
        for n in range(8):
            this_acc = np.sum(np.abs(scaled_coeffs[n::8]))
            vpu_acc_max = max(vpu_acc_max, this_acc)

        vpu_acc_headroom = vpu_acc_max / (2**39 - 1)

        if vpu_acc_headroom > 1:
            # accumulator can saturate, need to shift coeffs down
            vpu_acc_headroom_bits = -np.ceil(np.log2(vpu_acc_headroom))
            # shift the scaled coeffs
            scaled_coeffs *= 2**vpu_acc_headroom_bits
            shift += vpu_acc_headroom_bits

        # round the coeffs
        int_coeffs = np.round(scaled_coeffs).astype(int).tolist()

        return int_coeffs, int(shift)

    def reset_state(self) -> None:
        """Reset all the delay line values to zero."""
        self.buffer = np.zeros((self.n_chans, self.n_taps))
        self.buffer_int = [[0] * self.n_taps for _ in range(self.n_chans)]
        return

    def process(self, sample: float, channel: int = 0) -> float:
        """Update the buffer with the current sample and convolve with
        the filter coefficients, using floating point math.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on.

        Returns
        -------
        float
            The processed output sample.
        """
        # put new sample in buffer
        self.buffer[channel, self.buffer_idx[channel]] = sample
        this_idx = self.buffer_idx[channel]

        # decrement buffer so we point to the oldest sample
        if self.buffer_idx[channel] == 0:
            self.buffer_idx[channel] = self.n_taps - 1
        else:
            self.buffer_idx[channel] -= 1

        # do the convolution in two halves, [oldest:end] and [0:oldest]
        y = np.dot(self.buffer[channel, this_idx:], self.coeffs[: self.n_taps - this_idx])
        y += np.dot(self.buffer[channel, :this_idx], self.coeffs[self.n_taps - this_idx :])

        y = utils.saturate_float(y, self.Q_sig)

        return y

    def process_xcore(self, sample: float, channel: int = 0) -> float:
        """Update the buffer with the current sample and convolve with
        the filter coefficients, using int32 fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on.

        Returns
        -------
        float
            The processed output sample.
        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        # put new sample in buffer
        self.buffer_int[channel][self.buffer_idx[channel]] = sample_int
        this_idx = self.buffer_idx[channel]

        # decrement buffer so we point to the oldest sample
        if self.buffer_idx[channel] == 0:
            self.buffer_idx[channel] = self.n_taps - 1
        else:
            self.buffer_idx[channel] -= 1

        # do the convolution in two halves, [oldest:end] and [0:oldest]
        y = 0
        for n in range(self.n_taps - this_idx):
            y += utils.vpu_mult(self.buffer_int[channel][this_idx + n], self.coeffs_int[n])

        for n in range(this_idx):
            y += utils.vpu_mult(
                self.buffer_int[channel][n], self.coeffs_int[self.n_taps - this_idx + n]
            )

        # check accumulator hasn't overflown
        y = utils.int40(y)

        # shift accumulator
        if self.shift > 0:
            y += 1 << (self.shift - 1)
            y = y >> self.shift
        elif self.shift < 0:
            y = y << -self.shift

        # saturate
        y = utils.saturate_int64_to_int32(y)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt

    def freq_response(self, nfft: int = 32768) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the filter.

        Parameters
        ----------
        nfft : int
            Number of FFT points.

        Returns
        -------
        tuple
            A tuple containing the frequency values and the
            corresponding complex response.

        """
        w = np.fft.rfftfreq(nfft)
        h = np.fft.rfft(self.coeffs, nfft)
        return w, h
