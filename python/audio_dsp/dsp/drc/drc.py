# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32


FLT_MIN = np.finfo(float).tiny


class envelope_detector_peak(dspg.dsp_block):
    """
    Envelope detector that follows the absolute peak value of a signal.

    The attack time sets how fast the envelope detector ramps up. The
    release time sets how fast the envelope detector ramps down.

    Parameters
    ----------
    attack_t : float, optional
        Attack time of the envelope detector in seconds.
    release_t: float, optional
        Release time of the envelope detector in seconds.
    detect_t : float, optional
        Attack and relase time of the envelope detector in seconds. Sets
        attack_t == release_t. Cannot be used with attack_t or release_t
        inputs.

    Attributes
    ----------
    attack_alpha : float
        Attack time parameter used for exponential moving average in
        floating point processing.
    release_alpha : float
        Release time parameter used for exponential moving average in
        floating point processing.
    envelope : list[float]
        Current envelope value for each channel for floating point
        processing.
    attack_alpha_int : int
        attack_alpha in 32-bit int format.
    release_alpha_int : int
        release_alpha in 32-bit int format.
    envelope_int : list[int]
        current envelope value for each channel in 32-bit int format.

    """

    def __init__(
        self, fs, n_chans=1, attack_t=None, release_t=None, detect_t=None, Q_sig=dspg.Q_SIG
    ):
        super().__init__(fs, n_chans, Q_sig)

        if detect_t and (attack_t or release_t):
            ValueError("either detect_t OR (attack_t AND release_t) must be specified")
        elif detect_t:
            attack_t = detect_t
            release_t = detect_t

        # calculate EWM alpha from time constant
        self.attack_alpha, self.attack_alpha_int = drcu.alpha_from_time(attack_t, fs)
        self.release_alpha, self.release_alpha_int = drcu.alpha_from_time(release_t, fs)

        # initalise envelope state
        self.reset_state()

    def reset_state(self):
        """Reset the envelope to zero."""
        self.envelope = [0.0] * self.n_chans
        self.envelope_int = [utils.int32(0)] * self.n_chans

    def process(self, sample, channel=0):
        """
        Update the peak envelope for a signal, using floating point
        maths.

        Take one new sample and return the updated envelope. Input
        should be scaled with 0dB = 1.0.

        """
        if isinstance(sample, list) or isinstance(sample, np.ndarray):
            sample = sample[channel]

        sample_mag = abs(sample)

        # see if we're attacking or decaying
        if sample_mag > self.envelope[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1 - alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_xcore(self, sample, channel=0):
        """
        Update the peak envelope for a signal, using int32 fixed point
        maths.

        Take 1 new sample and return the updated envelope. If the input
        is int, return a int, otherwise expect float input
        and return float output.

        """
        if isinstance(sample, float):
            sample_int = utils.int32(round(sample * 2**self.Q_sig))
        elif (isinstance(sample, list) or isinstance(sample, np.ndarray)) and isinstance(
            sample[0], int
        ):
            sample_int = sample[channel]
        elif isinstance(sample, int):
            sample_int = sample
        else:
            raise TypeError("input must be float or int")

        sample_mag = abs(sample_int)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_int[channel]:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        acc = int(self.envelope_int[channel]) << 31
        mul = utils.int32(sample_mag - self.envelope_int[channel])
        acc += mul * alpha
        self.envelope_int[channel] = utils.int32_mult_sat_extract(acc, 1, 31)

        if isinstance(sample, float):
            return float(self.envelope_int[channel]) * 2**-self.Q_sig
        else:
            return self.envelope_int[channel]


class envelope_detector_rms(envelope_detector_peak):
    """
    Envelope detector that follows the RMS value of a signal.

    Note this returns the mean**2 value, there is no need to do the
    sqrt() as if the output is converted to dB, 10log10() can be taken
    instead of 20log10().

    The attack time sets how fast the envelope detector ramps up. The
    release time sets how fast the envelope detector ramps down.

    """

    def process(self, sample, channel=0):
        """
        Update the RMS envelope for a signal, using floating point
        maths.

        Take one new sample and return the updated envelope. Input
        should be scaled with 0dB = 1.0.

        Note this returns the mean**2 value, there is no need to do the
        sqrt() as if the output is converted to dB, 10log10() can be
        taken instead of 20log10().

        """
        # for rms use power
        sample_mag = sample**2

        # see if we're attacking or decaying
        if sample_mag > self.envelope[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1 - alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_xcore(self, sample, channel=0):
        """
        Update the RMS envelope for a signal, using int32 fixed point
        maths.

        Take one new sample and return the updated envelope. Input
        should be scaled with 0dB = 1.0.

        Note this returns the mean**2 value, there is no need to do the
        sqrt() as if the output is converted to dB, 10log10() can be
        taken instead of 20log10().

        """
        if isinstance(sample, float):
            sample_int = utils.int32(round(sample * 2**self.Q_sig))
        else:
            sample_int = sample

        acc = int(1 << (self.Q_sig - 1))
        acc += sample_int * sample_int
        sample_mag = utils.int32_mult_sat_extract(acc, 1, self.Q_sig)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_int[channel]:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        acc = int(self.envelope_int[channel]) << 31
        mul = utils.int32(sample_mag - self.envelope_int[channel])
        acc += mul * alpha
        self.envelope_int[channel] = utils.int32_mult_sat_extract(acc, 1, 31)

        # if we got floats, return floats, otherwise return ints
        if isinstance(sample, float):
            return float(self.envelope_int[channel]) * 2**-self.Q_sig
        else:
            return self.envelope_int[channel]


class compressor_limiter_base(dspg.dsp_block):
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

    # Limiter after Zolzer's DAFX & Guy McNally's "Dynamic Range Control
    # of Digital Audio Signals"
    def __init__(self, fs, n_chans, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.attack_alpha, self.attack_alpha_int = drcu.alpha_from_time(attack_t, fs)
        self.release_alpha, self.release_alpha_int = drcu.alpha_from_time(release_t, fs)

        # These are defined differently for peak and RMS limiters
        self.env_detector = None

        self.threshold = None
        self.threshold_int = None

        # slope is used for compressors, not limiters
        self.slope = None
        self.slope_f32 = None

        # initialise gain states
        self.reset_state()

        # set the gain calculation function handles
        self.gain_calc = None
        self.gain_calc_xcore = None

    def reset_state(self):
        """Reset the envelope detector to 0 and the gain to 1."""
        if self.env_detector:
            self.env_detector.reset_state()
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
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

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
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        acc = int(self.gain_int[channel]) << 31
        mul = utils.int32(new_gain_int - self.gain_int[channel])
        acc += mul * alpha
        self.gain_int[channel] = utils.int32_mult_sat_extract(acc, 1, 31)

        acc = 1 << 30
        acc += sample_int * self.gain_int[channel]
        y = utils.int32_mult_sat_extract(acc, 1, 31)

        return (
            (float(y) * 2**-self.Q_sig),
            (float(new_gain_int) * 2**-31),
            (float(envelope_int) * 2**-self.Q_sig),
        )

    def process_frame(self, frame):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process only take the first output.

        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process(this_chan[sample], channel=chan)[0]

        return output

    def process_frame_xcore(self, frame):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.
        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_xcore only take the first output.

        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_xcore(this_chan[sample], channel=chan)[0]

        return output


class limiter_peak(compressor_limiter_base):
    """
    A limiter based on the peak value of the signal. When the peak
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to it's original level
    after the envelope is below the threshold.

    Attributes
    ----------
    env_detector : envelope_detector_peak
        Nested peak envelope detector used to calculate the envelope of
        the signal.

    """

    def __init__(self, fs, n_chans, threshold_dB, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_dB)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_peak(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        # set the gain calculation function handles
        self.gain_calc = drcu.limiter_peak_gain_calc
        self.gain_calc_xcore = drcu.limiter_peak_gain_calc_xcore


class limiter_rms(compressor_limiter_base):
    """
    A limiter based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to it's original level
    after the envelope is below the threshold.


    Attributes
    ----------
    env_detector : envelope_detector_rms
        Nested RMS envelope detector used to calculate the envelope of
        the signal.
    threshold : float
        Value above which limiting occurs for floating point
        processing. Note the threshold is saves in the power domain, as
        the RMS envelope detector returns x**2

    """

    def __init__(self, fs, n_chans, threshold_dB, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_dB)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        # set the gain calculation function handles
        self.gain_calc = drcu.limiter_rms_gain_calc
        self.gain_calc_xcore = drcu.limiter_rms_gain_calc_xcore


class hard_limiter_peak(limiter_peak):
    def process(self, sample, channel=0):
        # do peak limiting
        y = super().process(sample, channel)

        # hard clip if above threshold
        if y > self.threshold:
            y = self.threshold
        if y < -self.threshold:
            y = -self.threshold
        return y

    # TODO process_int, super().process_int will return float though...
    def process_xcore(self, sample, channel=0):
        raise NotImplementedError


class soft_limiter_peak(limiter_peak):
    def __init__(
        self, fs, threshold_db, attack_t, release_t, delay=0, nonlinear_point=0.5, Q_sig=dspg.Q_SIG
    ):
        super().__init__(fs, threshold_db, attack_t, release_t, delay, Q_sig)
        self.nonlinear_point = nonlinear_point
        raise NotImplementedError

    # TODO soft clipping
    def process(self, sample, channel=0):
        raise NotImplementedError

    def process_xcore(self, sample, channel=0):
        raise NotImplementedError


class lookahead_limiter_peak(compressor_limiter_base):
    # peak limiter with built in delay for avoiding clipping
    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.env_detector = envelope_detector_peak(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.delay = np.ceil(attack_t * fs)
        self.delay_line = np.zeros(self.delay)
        raise NotImplementedError

    def process(self, sample, channel=0):
        raise NotImplementedError

    def process_xcore(self, sample, channel=0):
        raise NotImplementedError


class lookahead_limiter_rms(compressor_limiter_base):
    # rms limiter with built in delay for avoiding clipping
    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db_pow2gain(threshold_db)
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )
        self.delay = np.ceil(attack_t * fs)
        self.delay_line = np.zeros(self.delay)
        raise NotImplementedError

    def process(self, sample, channel=0):
        raise NotImplementedError

    def process_xcore(self, sample, channel=0):
        raise NotImplementedError


# TODO lookahead limiters and compressors
# TODO add soft limiter
# TODO add peak compressors
# TODO add soft knee compressors
# TODO add lookup compressors w/ some magic interface


class compressor_rms(compressor_limiter_base):
    """
    A compressor based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced by the compression ratio.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the comressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to it's original level after the envelope is below the
    threshold.

    Parameters
    ----------
    ratio : float
        Compression gain ratio applied when the signal is above the
        threshold

    Attributes
    ----------
    env_detector : envelope_detector_rms
        Nested RMS envelope detector used to calculate the envelope of
        the signal.
    ratio : float
        Compression gain ratio applied when the signal is above the
        threshold.
    slope : float
        The slope factor of the compressor, defined as
        `slope = (1 - 1/ratio)`.
    slope_f32 : float32
        The slope factor of the compressor, used for int32 to float32
        processing.
    threshold : float
        Value above which compression occurs for floating point
        processing.
    threshold_int : int
        Value above which compression occurs for int32 fixed point
        processing.

    """

    def __init__(
        self, fs, n_chans, ratio, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG
    ):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_db)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.slope = (1 - 1 / ratio) / 2.0
        self.slope_f32 = float32(self.slope)

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore


class noise_gate(compressor_limiter_base):
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
    attack_alpha : float
        The attack coefficient calculated from the release time and sample rate.
    release_alpha : float
        The release coefficient calculated from the attack time and sample rate.
    attack_alpha_int : int
        The attack coefficient as a 32-bit signed integer.
    release_alpha_int : int
        The release coefficient as a 32-bit signed integer.
    threshold : float
        The threshold below which the signal is gated.
    threshold_int : int
        The threshold level as a 32-bit signed integer.
    env_detector : envelope_detector_peak
        An instance of the envelope_detector_peak class used for envelope detection.

    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        # don't use super(), as the attack and release times are swapped
        # but still init the block
        dspg.dsp_block.__init__(self, fs, n_chans, Q_sig)

        # for the noise gate, the attack and release times are swapped
        # i.e. attack time is after going under threshold instead of over
        # the device will expect these swapped back when doing control
        self.release_alpha, self.release_alpha_int = drcu.alpha_from_time(attack_t, fs)
        self.attack_alpha, self.attack_alpha_int = drcu.alpha_from_time(release_t, fs)

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

        # slope is used for compressors, not noise gates
        self.slope = None
        self.slope_f32 = None

        self.reset_state()

    def reset_state(self):
        """Reset the envelope detector to 1 and the gain to 1, so the
        gate starts off."""
        self.env_detector.envelope = [1] * self.n_chans
        self.env_detector.envelope_int = [utils.int32(2**self.Q_sig)] * self.n_chans
        self.gain = [1] * self.n_chans
        self.gain_int = [2**31 - 1] * self.n_chans


if __name__ == "__main__":
    # import audio_dsp.dsp.signal_gen as gen

    fs = 48000
    x1 = np.ones(int(fs * 0.5))
    x2 = 0.1 * np.ones(int(fs * 0.5))

    x = np.concatenate((x1, x2, x1, x2))
    # x = gen.sin(fs, 0.2, 997*4, 1)

    t = np.arange(len(x)) / fs

    threshold = -6
    at = 0.01

    lt = limiter_rms(fs, 1, threshold, at, 0.3)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

    lt.reset_state()

    y_int = np.zeros_like(x)
    f_int = np.zeros_like(x)
    env_int = np.zeros_like(x)

    # import cProfile

    # with cProfile.Profile() as pr:
    for n in range(len(y)):
        y_int[n], f_int[n], env_int[n] = lt.process_xcore(x[n])
        # pr.print_stats(sort='time')

    thresh_passed = np.argmax(utils.db(env) > threshold)
    sig_3dB = np.argmax(utils.db(y) < (threshold + 3))

    measured_at = t[sig_3dB] - t[thresh_passed]
    print(measured_at)

    import matplotlib.pyplot as plt

    # plt.plot(t, utils.db(x))
    plt.plot(t, utils.db(y))
    plt.plot(t, utils.db(env))
    # plt.plot(t, utils.db(f))
    plt.plot(t, utils.db(y_int))
    plt.plot(t, utils.db(env_int))

    # plt.legend(["x", "y", "env", "gain"])
    plt.legend(["y", "env", "y_int", "env_int"])
    plt.grid()
    plt.show()

    pass
