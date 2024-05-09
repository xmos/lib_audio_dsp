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


# temporary mess
this_dir = os.path.dirname(os.path.abspath((__file__)))
sys.path.append(str(Path(this_dir, "..", "..", "..", "..", "lib_xcore_math", "lib_xcore_math", "script").resolve()))

from gen_fir_filter_s32 import find_filter_parameters


class fir_direct(dspg.dsp_block):
    """An FIR filter, implemented in direct form in the time domain

    """

    def __init__(self, fs: float, n_chans: int, coeffs_path: Path, coeff_scaling: str="none", Q_sig: int = dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        raw_coeffs = np.loadtxt(coeffs_path)
        self.taps = len(raw_coeffs)

        if coeff_scaling is None or coeff_scaling.lower() == "none":
            pass
        elif coeff_scaling.lower() == "unity_gain":
            self.scale_coeffs_unity_gain(coeffs)
        elif coeff_scaling.lower() == "never_clip":
            pass
        else:
            raise ValueError("Unknown coeff_scaling requested")


        self.coeffs, self.coeffs_int, self.shift, self.exponent_diff, self.n_taps = self.get_coeffs(coeffs_path)
        self.buffer = np.zeros((self.n_chans, self.n_taps))
        self.buffer_int = [[0]*self.n_taps]*self.n_chans
        self.buffer_idx = [0] * self.n_chans
        self.buffer_idx_int = [0] * self.n_chans

    def scale_coeffs_unity_gain(self, coeffs):
        coeff_sum = np.sum(coeffs)
        coeffs /= coeff_sum
        return coeffs

    def scale_coeffs_never_clip(self, coeffs):
        coeff_sum = np.sum(np.abs(coeffs))

        pass
        return

    def check_coeff_scaling(self, coeffs):
        headroom = 2**(31 - self.Q_sig)
        coeff_sum = np.sum(np.abs(coeffs))

        if coeff_sum > headroom:
            warnings.warn("Headroom of %d dB is not sufficient to guarentee no clipping." % utils.db(headroom))

        return

    def make_int_coeffs(self, coeffs):
        # check headroom on coefficients, preparing for multiplicaiton
        # by int32_max
        max_coeff = np.max(coeffs)
        max_coeff_headroom = -np.ceil(np.log2(max_coeff))


        # we have 43 bits in the accumulator (from summing 8x 40b accs),
        # so can be 12 bits
        coeff_sum = np.sum(np.abs(coeffs))
        coeff_sum_headroom = -np.ceil(np.log2(coeff_sum))


    def get_coeffs(self, coeffs_path):
        coeffs = np.loadtxt(coeffs_path)
        taps = len(coeffs)
        args = types.SimpleNamespace()
        args.filter_coefficients = coeffs
        args.input_headroom = 0
        args.output_headroom = 0
        scaled_coefs_s32, shift, exponent_diff = find_filter_parameters(args)

        # coeffs = (coeffs)
        int_coeffs = (scaled_coefs_s32).tolist()
        return coeffs, int_coeffs, shift, exponent_diff, taps

    def reset_state(self) -> None:
        """Reset all the delay line values to zero."""
        self.buffer = np.zeros((self.n_chans, self.n_taps))
        self.buffer_int = [[0]*self.n_taps]*self.n_chans
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

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt

    def freq_response(self, nfft: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the gain, assumed to be a
        flat response scaled by the gain.

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
        h = np.ones_like(w) * self.gain
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