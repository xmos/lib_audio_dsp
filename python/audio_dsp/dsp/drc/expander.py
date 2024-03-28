# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32
from audio_dsp.dsp.drc import envelope_detector_peak, compressor_limiter_base

FLT_MIN = np.finfo(float).tiny

class expander_base(compressor_limiter_base):
    """
    A base class shared by compressor and limiter objects.

    The compressor and limiter have very similar structures, with
    differences in the gain calculation. All the shared code and
    parameters are calculated in this base class.

    Parameters
    ----------
    n_chans : int
        number of parallel channels the compressor/limiter runs on. The
        channels are limited/compressed separately, only the constant
        parameters are shared.
    attack_t : float, optional
        Attack time of the compressor/limiter in seconds.
    release_t: float, optional
        Release time of the compressor/limiter in seconds.

    Attributes
    ----------
    env_detector : envelope_detector_peak
        Nested envelope detector used to calculate the envelope of the
        signal. Either a peak or RMS envelope detector can be used.
    threshold : float
        Value above which comression/limiting occurs for floating point
        processing.
    gain : list[float]
        Current gain to be applied to the signal for each channel for
        floating point processing.
    attack_alpha : float
        Attack time parameter used for exponential moving average in
        floating point processing.
    release_alpha : float
        Release time parameter used for exponential moving average in
        floating point processing.
    threshold_int : int
        Value above which comression/limiting occurs for int32 fixed
        point processing.
    gain_int : list[int]
        Current gain to be applied to the signal for each channel for
        int32 fixed point processing.
    attack_alpha_int : int
        attack_alpha in 32-bit int format.
    release_alpha_int : int
        release_alpha in 32-bit int format.

    """

    # The expander is based on the limiter, but with the attack and
    # release in the gain calulation swapped (i.e. release after going 
    # above the threshold)

    def reset_state(self):
        """Reset the envelope detector to 1 and the gain to 1, so the
        gate starts off."""
        if self.env_detector is not None:
            self.env_detector.envelope = [1] * self.n_chans
            self.env_detector.envelope_int = [utils.int32(2**self.Q_sig)] * self.n_chans
        self.gain = [1] * self.n_chans
        self.gain_int = [2**31 - 1] * self.n_chans

    def process(self, sample, channel=0):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for compression/limiting, using floating point
        maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        # get envelope from envelope detector
        envelope = self.env_detector.process(sample, channel)  # type: ignore
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # calculate the gain, this function should be defined by the
        # child class
        new_gain = self.gain_calc(envelope, self.threshold, self.slope)  # type: ignore

        # see if we're attacking or decaying
        if new_gain < self.gain[channel]:
            alpha = self.release_alpha
        else:
            alpha = self.attack_alpha

        # do exponential moving average
        self.gain[channel] = ((1 - alpha) * self.gain[channel]) + (alpha * new_gain)

        # apply gain to input
        y = self.gain[channel] * sample
        return y, new_gain, envelope

    def process_xcore(self, sample, channel=0):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for compression/limiting, using int32 fixed point
        maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        sample_int = utils.int32(round(sample * 2**self.Q_sig))
        # get envelope from envelope detector
        envelope_int = self.env_detector.process_xcore(sample_int, channel)
        # avoid /0
        envelope_int = max(envelope_int, 1)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain_int = self.gain_calc_xcore(envelope_int, self.threshold_int, self.slope_f32)

        # see if we're attacking or decaying
        if new_gain_int < self.gain_int[channel]:
            alpha = self.release_alpha_int
        else:
            alpha = self.attack_alpha_int

        # do exponential moving average
        self.gain_int[channel] = drcu.calc_ema_xcore(self.gain_int[channel], new_gain_int, alpha)

        # apply gain
        y = drcu.apply_gain_xcore(sample_int, self.gain_int[channel])

        return (
            (float(y) * 2**-self.Q_sig),
            (float(new_gain_int) * 2**-self.Q_alpha),
            (float(envelope_int) * 2**-self.Q_sig),
        )


class noise_gate(expander_base):
    """A noise gate that reduces the level of an audio signal when it
    falls below a threshold.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced to 0 (based on the release time). When the
    envelope returns above the threshold, the gain applied to the signal
    is increased to 1 over the attack time.

    The initial state of the noise gate is with the gate open (no
    attenuation), assuming a full scale signal has been present before
    t = 0.

    Parameters
    ----------
    threshold_db : float
        The threshold level in decibels below which the audio signal is
        attenuated.

    Attributes
    ----------
    threshold : float
        The threshold below which the signal is gated.
    threshold_int : int
        The threshold level as a 32-bit signed integer.
    env_detector : envelope_detector_peak
        An instance of the envelope_detector_peak class used for envelope detection.

    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_peak(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        # set the gain calculation function handles
        self.gain_calc = drcu.noise_gate_gain_calc
        self.gain_calc_xcore = drcu.noise_gate_gain_calc_xcore

        self.reset_state()


class noise_suppressor(expander_base):
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
    threshold_db : float
        The threshold level in decibels below which the audio signal is
        attenuated.

    Attributes
    ----------
    threshold : float
        The threshold below which the signal is gated.
    threshold_int : int
        The threshold level as a 32-bit signed integer.
    env_detector : envelope_detector_peak
        An instance of the envelope_detector_peak class used for envelope detection.

    """

    def __init__(self, fs, n_chans, ratio, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.threshold_int = max(1, self.threshold_int)
        # self.inv_threshold_int = utils.int32(1/self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_peak(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.slope = (1 - ratio)
        self.slope_f32 = float32(self.slope)

        # set the gain calculation function handles
        self.gain_calc = drcu.noise_suppressor_gain_calc
        self.gain_calc_xcore = drcu.noise_suppressor_gain_calc_xcore

        self.reset_state()



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    ns = noise_suppressor(48000, 1, 3, -20, 0.01, 0.1)
    ing, outg = ns.get_gain_curve()

    plt.plot(ing, outg)
    plt.axis('equal')
    plt.xlim([ing[0], ing[-1]])
    plt.ylim([ing[0], ing[-1]])
    plt.grid()
    plt.show()


