# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for simple filters."""

import audio_dsp.dsp.generic as dspg
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.utils as utils
from copy import deepcopy
import warnings
import audio_dsp.dsp.reverb_base as rvb
import audio_dsp.dsp.reverb as rv


class lowpass_1ord(dspg.dsp_block):
    """A first order lowpass filter. This is a simple exponential moving
    average.

    Parameters
    ----------
    bandwidth : float
        Set the low pass bandwidth.
    """

    def __init__(self, fs, n_chans, bandwidth, Q_sig=dspg.Q_SIG):
        assert n_chans == 1
        super().__init__(fs, n_chans, Q_sig)

        self._filterstore = 0.0
        self._filterstore_int = 0
        self.set_bandwidth(bandwidth)

    def set_bandwidth(self, bandwidth):
        """Set the bandwith of the low pass filter, as a ratio. Higher
        bandwidth will result in a higher filter cutoff frequency.
        """
        self.coeff_1 = bandwidth
        self.coeff_2 = 1 - self.coeff_1
        # super critical these add up, but also don't overflow int32...
        self.coeff_1_int = max(utils.int32(self.coeff_1 * 2**rvb.Q_VERB - 1), 1)
        self.coeff_2_int = utils.int32((2**31 - 1) - self.coeff_1_int + 1)

    def reset_state(self):
        """Reset all the filterstore values to zero."""
        self._filterstore = 0.0
        self._filterstore_int = 0

    def process(self, sample):  # type: ignore : overloads base class
        """
        Apply a low pass filter to a signal, using floating point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 1.0.

        """
        output = (sample * self.coeff_1) + (self._filterstore * self.coeff_2)

        self._filterstore = output

        return output

    def process_xcore(self, sample_int):  # type: ignore : overloads base class
        """
        Apply a low pass filter to a signal, using fixed point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 2**Q_SIG.

        Parameters
        ----------
        sample_int : int
            Input sample as an integer.

        """
        assert isinstance(sample_int, int), "Input sample must be an integer"

        # do state calculation in int64 accumulator so we only quantize once
        output = utils.int64(
            sample_int * self.coeff_1_int + self._filterstore_int * self.coeff_2_int
        )
        output = rvb.scale_sat_int64_to_int32_floor(output)
        self._filterstore_int = output

        return output


class allpass(dspg.dsp_block):
    """An Nth order all-pass filter, with flat frequency response but varying
    phase response. The delay sets the order of the all-pass filter,
    with 90 degrees of phase shift per order at f0 = fs/(2*N). The feedback
    gain controls the steepness of the phase shift.

    Parameters
    ----------
    max_delay : int
        Maximum delay of the all-pass.
    feedback_gain : float
        Gain applied to the delayed feedback path in the all-pass. Sets
        the reverb time.
    """

    def __init__(self, fs, n_chans, max_delay, feedback_gain, Q_sig=dspg.Q_SIG):
        assert n_chans == 1
        super().__init__(fs, n_chans, Q_sig)

        # max delay cannot be changed, or you'll overflow the buffer
        self._max_delay = max_delay
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay

        self.delay = max_delay
        self.feedback = feedback_gain

        self._buffer_idx = 0

    @property
    def feedback(self):
        """Allpass gain coefficient."""
        return self._feedback

    @feedback.setter
    def feedback(self, x):
        self._feedback = x
        if self.feedback < 0:
            # if negative, ensure we cap to -INT32_MAX
            self.feedback_int = -rvb.float_to_q_verb(-self.feedback)
        else:
            self.feedback_int = rvb.float_to_q_verb(self.feedback)

    def set_delay(self, delay):
        """Set the length of the delay line. Will saturate to max_delay."""
        if delay <= self._max_delay:
            self.delay = delay
        else:
            self.delay = self._max_delay
            warnings.warn(
                "Delay cannot be greater than max delay, setting to max delay", UserWarning
            )

    def reset_state(self):
        """Reset all the delay line values to zero."""
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay
        return

    def process(self, sample):  # type: ignore : overloads base class
        """
        Apply an all pass filter to a signal, using floating point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 1.0.

        """
        output = sample
        buff_out = self._buffer[self._buffer_idx]
        output -= buff_out * self.feedback

        self._buffer[self._buffer_idx] = output

        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        output = buff_out + output * self.feedback

        return output

    def process_xcore(self, sample_int):  # type: ignore : overloads base class
        """
        Apply an all pass filter to a signal, using fixed point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 2**Q_SIG.

        Parameters
        ----------
        sample_int : int
            Input sample as an integer.

        """
        assert isinstance(sample_int, int), "Input sample must be an integer"

        buff_out = self._buffer_int[self._buffer_idx]

        # do buffer calculation in int64 accumulator so we only quantize once
        output = utils.int64((sample_int << rvb.Q_VERB) - buff_out * self.feedback_int)
        output = rvb.scale_sat_int64_to_int32_floor(output)

        self._buffer_int[self._buffer_idx] = output

        # move buffer head
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        output = utils.int64((buff_out << rvb.Q_VERB) + output * self.feedback_int)
        output = rvb.scale_sat_int64_to_int32_floor(output)

        return output
