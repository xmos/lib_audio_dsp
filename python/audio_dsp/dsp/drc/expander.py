# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The expander DSP blocks."""

import numpy as np

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32
from audio_dsp.dsp.drc.drc import compressor_limiter_base, peak_compressor_limiter_base

FLT_MIN = np.finfo(float).tiny


class expander_base(compressor_limiter_base):
    """
    A base class for expanders (including noise suppressors).

    Expanders differ from compressors in that they reduce the level of a
    signal when it falls below a threshold (instead of above). The
    attack time is still defined as how quickly the gain is changed
    after the envelope exceeds the threshold.

    Expanders, noise gates and noise suppressors have very similar
    structures, with differences in the gain calculation. All the shared
    code and parameters are calculated in this base class.

    Parameters
    ----------
    n_chans : int
        number of parallel channels the expander runs on. The
        channels are expanded separately, only the constant
        parameters are shared.
    threshold_db : float
        Threshold in decibels below which expansion occurs. This cannot
        be greater than the maximum value representable in
        Q_SIG format, and will saturate to that value.
    attack_t : float
        Attack time of the expander in seconds. This cannot be
        faster than 2/fs seconds, and saturates to that
        value. Exceptionally large attack times may converge to zero.
    release_t: float
        Release time of the expander in seconds. This cannot
        be faster than 2/fs seconds, and saturates to that
        value. Exceptionally large release times may converge to zero.

    Attributes
    ----------
    threshold : float
        Value below which expanding occurs for floating point
        processing.
    threshold_int : int
        Value below which expanding occurs for int32 fixed
        point processing.

    """

    def reset_state(self):
        """Reset the envelope detector to 1 and the gain to 1, so the
        gate starts off.
        """
        if self.env_detector is not None:
            self.env_detector.envelope = [1] * self.n_chans
            self.env_detector.envelope_int = [utils.int32(2**self.Q_sig - 1)] * self.n_chans
        self.gain = [1] * self.n_chans
        self.gain_int = [2**31 - 1] * self.n_chans

    def process(self, sample, channel=0):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for expanding, using floating point
        maths.

        Take one new sample and return the expanded sample.
        Input should be scaled with 0 dB = 1.0.

        """
        # get envelope from envelope detector
        envelope = self.env_detector.process(sample, channel)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # calculate the gain, this function should be defined by the
        # child class
        new_gain = self.gain_calc(envelope, self.threshold, self.slope)  # type: ignore : base inits to None

        # see if we're attacking or decaying
        if new_gain < self.gain[channel]:
            # below threshold, gain < unity
            alpha = self.release_alpha
        else:
            # above threshold, gain = unity
            alpha = self.attack_alpha

        # do exponential moving average
        self.gain[channel] = ((1 - alpha) * self.gain[channel]) + (alpha * new_gain)

        # apply gain to input
        y = self.gain[channel] * sample
        return y, new_gain, envelope

    def process_xcore(self, sample, channel=0, return_int=False):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for expanding, using int32 fixed point
        maths.

        Take one new sample and return the expanded sample.
        Input should be scaled with 0 dB = 1.0.

        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)
        # get envelope from envelope detector
        envelope_int = self.env_detector.process_xcore(sample_int, channel)
        # avoid /0
        envelope_int = max(envelope_int, 1)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain_int = self.gain_calc_xcore(envelope_int, self.threshold_int, self.slope_f32)  # pyright: ignore : base inits to None

        # see if we're attacking or decaying
        if new_gain_int < self.gain_int[channel]:
            # below threshold, gain < unity
            alpha = self.release_alpha_int
        else:
            # above threshold, gain = unity
            alpha = self.attack_alpha_int

        # do exponential moving average
        self.gain_int[channel] = drcu.calc_ema_xcore(self.gain_int[channel], new_gain_int, alpha)

        # apply gain
        y = drcu.apply_gain_xcore(sample_int, self.gain_int[channel])

        if return_int:
            return y, new_gain_int, envelope_int
        else:
            return (
                utils.int32_to_float(y, self.Q_sig),
                utils.int32_to_float(new_gain_int, self.Q_alpha),
                utils.int32_to_float(envelope_int, self.Q_sig),
            )


class peak_expander_base(expander_base, peak_compressor_limiter_base):
    """A generic expander base class that uses a peak envelope detector.

    Inheritance from expander_base is prioritised over
    peak_compressor_limiter_base due to the order in the definition. To
    confirm this, peak_expander_base.__mro__ can be inspected.
    """


class noise_gate(peak_expander_base):
    """A noise gate that reduces the level of an audio signal when it
    falls below a threshold.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced to 0 (based on the release time). When the
    envelope returns above the threshold, the gain applied to the signal
    is increased to 1 over the attack time.

    The initial state of the noise gate is with the gate open (no
    attenuation), assuming a full scale signal has been present before
    t = 0.
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        # set the gain calculation function handles
        self.gain_calc = drcu.noise_gate_gain_calc
        self.gain_calc_xcore = drcu.noise_gate_gain_calc_xcore


class noise_suppressor_expander(peak_expander_base):
    """A noise suppressor that reduces the level of an audio signal when
    it falls below a threshold. This is also known as an expander.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced relative to the expansion ratio over the
    release time. When the envelope returns above the threshold, the
    gain applied to the signal is increased to 1 over the attack time.

    The initial state of the noise suppressor is with the suppression
    off, assuming a full scale signal has been present before
    t = 0.

    Parameters
    ----------
    ratio : float
        The expansion ratio applied to the signal when the envelope
        falls below the threshold.

    Attributes
    ----------
    ratio : float
    slope : float
        The slope factor of the expander, defined as
        `slope = 1 - ratio`.
    slope_f32 : float32
        The slope factor of the expander, used for int32 to float32
        processing.
    """

    def __init__(self, fs, n_chans, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        # property calculates the slopes as well
        self.ratio = ratio

        # set the gain calculation function handles
        self.gain_calc = drcu.noise_suppressor_expander_gain_calc
        self.gain_calc_xcore = drcu.noise_suppressor_expander_gain_calc_xcore

    @property
    def ratio(self):
        """Expansion gain ratio applied when the signal is below the
        threshold; changing this property also updates the slope used in
        the fixed and floating point implementation.
        """
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value
        self.slope, self.slope_f32 = drcu.peak_expander_slope_from_ratio(self.ratio)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nse = noise_suppressor_expander(48000, 1, 3, -20, 0.01, 0.1)
    ing, outg = nse.get_gain_curve()

    plt.plot(ing, outg)
    plt.axis("equal")
    plt.xlim([ing[0], ing[-1]])
    plt.ylim([ing[0], ing[-1]])
    plt.grid()
    plt.show()
