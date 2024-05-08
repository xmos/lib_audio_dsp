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

    def __init__(self, fs: float, n_chans: int, coeffs_path: Path, Q_sig: int = dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.coeffs, self.coeffs_int, self.shift, self.exponent_diff, self.n_taps = self.get_coeffs(coeffs_path)
        self.buffer = np.zeros((self.n_chans, self.n_taps))
        self.buffer_int = [[0]*self.n_taps]*self.n_chans
        self.buffer_idx = [0] * self.n_chans
        self.buffer_idx_int = [0] * self.n_chans

    def get_coeffs(self, coeffs_path):
        coeffs = np.loadtxt(coeffs_path)
        taps = len(coeffs)
        args = types.SimpleNamespace()
        args.filter_coefficients = coeffs
        args.input_headroom = 0
        args.output_headroom = 0
        scaled_coefs_s32, shift, exponent_diff = find_filter_parameters(args)

        coeffs = np.flip(coeffs)
        return coeffs, scaled_coefs_s32, shift, exponent_diff, taps
        pass

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

        # increment buffer so we point to the oldest sample
        self.buffer_idx[channel] += 1
        if self.buffer_idx[channel] >= self.n_taps:
            self.buffer_idx[channel] = 0

        this_idx = self.buffer_idx[channel]

        # do the convolution in two halves, [oldest:end] and [0:oldest]
        y = np.dot(self.buffer[channel, this_idx:], self.coeffs[:self.n_taps-this_idx])
        y += np.dot(self.buffer[channel, :this_idx], self.coeffs[self.n_taps-this_idx:])


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
        # for rounding
        acc = 1 << (Q_GAIN - 1)
        acc += sample_int * self.gain_int
        y = utils.int32_mult_sat_extract(acc, 1, Q_GAIN)

        y_flt = float(y) * 2**-self.Q_sig

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