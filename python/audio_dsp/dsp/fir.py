# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import warnings
from pathlib import Path
import sys
import os
import types

from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils


class fir_direct(dspg.dsp_block):
    """An FIR filter, implemented in direct form in the time domain

    """

    def __init__(self, fs: float, n_chans: int, coeffs_path: Path, Q_sig: int = dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        raw_coeffs = np.loadtxt(coeffs_path)
        self.taps = len(raw_coeffs)

        self.coeffs = np.loadtxt(coeffs_path)
        self.n_taps = len(self.coeffs)
        self.coeffs_int, self.shift = self.check_coeff_scaling(self.coeffs)

        # self.coeffs, self.coeffs_int, self.shift, self.exponent_diff, self.n_taps = self.get_coeffs(coeffs_path)
        self.reset_state()
        self.buffer_idx = [0] * self.n_chans
        self.buffer_idx_int = [0] * self.n_chans

    def check_coeff_scaling(self, coeffs):
        
        int32_max = 2**31 - 1

        # scale to Q30, to match VPU shift but keep as double for now
        # until we see how many bits we have
        scaled_coeffs = coeffs * (2**30)

        # find how many bits we can (or need to) shift the coeffs by
        max_coeff = np.max(np.abs(scaled_coeffs))
        coeff_headroom = max_coeff/int32_max
        coeff_headroom_bits = -np.ceil(np.log2(coeff_headroom))
        shift = coeff_headroom_bits

        # shift the scaled coeffs
        scaled_coeffs *= 2**coeff_headroom_bits

        # check the gain of the filter will fit in the output Q format
        headroom = utils.db(2**(31 - self.Q_sig))
        w, h = self.freq_response()
        coeff_max_gain = np.max(utils.db(h))

        if coeff_max_gain > headroom:
            warnings.warn("Headroom of %d dB is not sufficient to guarentee no clipping." % (headroom))

        # VPU stripes the convolution across 8 40b accumulators
        vpu_acc_max = 0
        for n in range(8):
            this_acc = np.sum(np.abs(scaled_coeffs[n::8]))
            vpu_acc_max = max(vpu_acc_max, this_acc)

        vpu_acc_headroom = vpu_acc_max/(2**39 - 1)

        if vpu_acc_headroom > 1:
            # accumulator can saturate, need to shift coeffs down
            vpu_acc_headroom_bits = -np.ceil(np.log2(vpu_acc_headroom))
            # shift the scaled coeffs
            scaled_coeffs *= 2**vpu_acc_headroom_bits
            shift += vpu_acc_headroom_bits

        # round the coeffs
        int_coeffs = np.round(scaled_coeffs).astype(int).tolist()

        return int_coeffs, int(shift)

    def make_int_coeffs(self, coeffs):
        # check headroom on coefficients, preparing for multiplicaiton
        # by int32_max
        max_coeff = np.max(coeffs)
        max_coeff_headroom = -np.ceil(np.log2(max_coeff))


        # we have 43 bits in the accumulator (from summing 8x 40b accs),
        # so can be 12 bits
        coeff_sum = np.sum(np.abs(coeffs))
        coeff_sum_headroom = -np.ceil(np.log2(coeff_sum))

    def reset_state(self) -> None:
        """Reset all the delay line values to zero."""
        self.buffer = np.zeros((self.n_chans, self.n_taps))
        self.buffer_int = [[0]*self.n_taps for _ in range(self.n_chans)]
        return

    def process(self, sample: float, channel: int = 0) -> float:
        """Update the buffer with the current sample and convolve with
        the filter coefficients, using floating point math

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
        y = np.dot(self.buffer[channel, this_idx:], self.coeffs[:self.n_taps-this_idx])
        y += np.dot(self.buffer[channel, :this_idx], self.coeffs[self.n_taps-this_idx:])

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
        for n in range(self.n_taps-this_idx):
            y += utils.vpu_mult(self.buffer_int[channel][this_idx + n], self.coeffs_int[n])

        for n in range(this_idx):
            y += utils.vpu_mult(self.buffer_int[channel][n], self.coeffs_int[self.n_taps - this_idx + n])

        # check accumulator hasn't overflown
        y = utils.int40(y)

        # shift accumulator
        if self.shift > 0:
            y += (1 << (self.shift - 1))
            y = y >> self.shift
        elif self.shift < 0:
            y = y << -self.shift

        # saturate
        y = utils.saturate_int64_to_int32(y)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt

    def freq_response(self, nfft: int = 32768) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the filter

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


if __name__ == "__main__":

    coeffs = np.zeros(1000)
    coeffs[0] = 1

    np.savetxt("simple_filter.txt", coeffs)

    fir_test = fir(48000, 1, "simple_filter.txt")

    from audio_dsp.dsp.signal_gen import pink_noise

    signal = pink_noise(48000, 1, 0.5)
    
    out_flt = np.zeros_like(signal)

    for n in range(len(signal)):
        out_flt[n] = fir_test.process(signal[n])

    np.testing.assert_equal(signal, out_flt)