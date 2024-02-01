import numpy as np
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg


class envelope_detector_peak(dspg.dsp_block):
    """
    Envelope detector that follows the absolute peak value of a signal

    The attack time sets how fast the envelope detector ramps up. The decay
    time sets how fast the envelope detector ramps down.

    Parameters
    ----------
    fs : int
        sampling frequency in Hz.
    attack_t : float, optional
        Attack time of the envelope detector in seconds.
    release_t: float, optional
        Release time of the envelope detector in seconds.
    detect_t : float, optional
        Attack and relase time of the envelope detector in seconds. Sets
        attack_t == release_t Cannot be used with attack_t or release_t inputs.
    Q_sig: int, optional
        Q format of the signal, number of bits after the decimal point.
        Defaults to Q27.

    Attributes
    ----------
    attack_alpha : float
        Attack time parameter used for exponential moving average in floating
        point processing.
    release_alpha : float
        Release time parameter used for exponential moving average in floating
        point processing.
    envelope : float
        Current envelope value for floating point processing.
    attack_alpha_uq30 : int
        attack_alpha in unsigned Q2.30 format.
    release_alpha_uq30 : int
        release_alpha in unsigned Q2.30 format.
    envelope_s32 : float_s32
        current envelope value in float_s32 format.

    """

    def __init__(self, fs, attack_t=None, release_t=None, detect_t=None, Q_sig=dspg.Q_SIG):
        super().__init__(fs, Q_sig)

        if detect_t and (attack_t or release_t):
            ValueError("either detect_t OR (attack_t AND release_t) must be specified")
        elif detect_t:
            attack_t = detect_t
            release_t = detect_t

        # attack times simplified from McNally, seem pretty close. Assumes the
        # time constant of a digital filter is the -3dB point where
        # abs(H(z))**2 = 0.5.
        T = 1/fs
        # attack/release time can't be faster than the length of 2 samples.
        self.attack_alpha = min(2*T / attack_t, 1.0)
        self.release_alpha = min(2*T / release_t, 1.0)
        self.envelope = 0

        # very long times might quantize to zero, maybe just limit a better way
        assert self.attack_alpha > 0
        assert self.release_alpha > 0

        self.attack_alpha_uq30 = utils.uq_2_30(round(self.attack_alpha * 2**30))
        self.release_alpha_uq30 = utils.uq_2_30(round(self.release_alpha * 2**30))
        self.envelope_s32 = utils.float_s32([0, -self.Q_sig])

        # very long times might quantize to zero
        assert self.attack_alpha_uq30 > 0
        assert self.attack_alpha_uq30 > 0

    def reset_state(self):
        """Reset the envelope to zero."""
        self.envelope = 0
        self.envelope_s32 = utils.float_s32(0, -self.Q_sig)

    def process(self, sample):
        """
        Update the peak envelope for a signal, using floating point maths.

        Take 1 new sample and return the updated envelope. Input should be
        scaled with 0dB = 1.0.

        """
        sample_mag = abs(sample)

        # see if we're attacking or decaying
        if sample_mag > self.envelope:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope = ((1-alpha) * self.envelope) + (alpha * sample_mag)

        return self.envelope

    def process_int(self, sample):
        """
        Update the peak envelope for a signal, using float_s32 maths.

        Take 1 new sample and return the updated envelope. If the input is
        float_s32, return a float_s32, otherwise expect float input and return
        float output.

        """
        if isinstance(sample, utils.float_s32):
            # don't do anything if we got float_s32, this function was probably
            # called from a limiter or compressor
            sample_s32 = sample
        else:
            # if input isn't float_s32, convert it
            sample_s32 = utils.float_s32(sample)

        sample_mag = abs(sample_s32)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_s32:
            alpha = self.attack_alpha_uq30
        else:
            alpha = self.release_alpha_uq30

        # do exponential moving average
        self.envelope_s32 = utils.float_s32_ema(sample_mag, self.envelope_s32, alpha)

        # if we got floats, return floats, otherwise return float_s32
        if isinstance(sample, utils.float_s32):
            return self.envelope_s32
        else:
            return float(self.envelope_s32)


class envelope_detector_rms(envelope_detector_peak):
    """
    Envelope detector that follows the RMS value of a signal.

    Note this returns the mean**2 value, there is no need to do the sqrt() as
    if the output is converted to dB, 10log10() can be taken instead of
    20log10().

    The attack time sets how fast the envelope detector ramps up. The decay
    time sets how fast the envelope detector ramps down.

    Parameters
    ----------
    fs : int
        sampling frequency in Hz.
    attack_t : float, optional
        Attack time of the envelope detector in seconds.
    release_t: float, optional
        Release time of the envelope detector in seconds.
    detect_t : float, optional
        Attack and relase time of the envelope detector in seconds. Sets
        attack_t == release_t Cannot be used with attack_t or release_t inputs.
    Q_sig: int, optional
        Q format of the signal, number of bits after the decimal point.
        Defaults to Q27.

    Attributes
    ----------
    attack_alpha : float
        Attack time parameter used for exponential moving average in floating
        point processing.
    release_alpha : float
        Release time parameter used for exponential moving average in floating
        point processing.
    envelope : float
        Current mean squared envelope value for floating point processing.
    attack_alpha_uq30 : int
        attack_alpha in unsigned Q2.30 format.
    release_alpha_uq30 : int
        release_alpha in unsigned Q2.30 format.
    envelope_s32 : float_s32
        current mean squared envelope value in float_s32 format.

    """

    def process(self, sample):
        """
        Update the RMS envelope for a signal, using floating point maths.

        Take 1 new sample and return the updated envelope. Input should be
        scaled with 0dB = 1.0.

        Note this returns the mean**2 value, there is no need to do the sqrt() as
        if the output is converted to dB, 10log10() can be taken instead of
        20log10().

        """
        # for rms use power
        sample_mag = sample**2

        # see if we're attacking or decaying
        if sample_mag > self.envelope:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope = ((1-alpha) * self.envelope) + (alpha * sample_mag)

        return self.envelope

    def process_int(self, sample):
        """
        Update the RMS envelope for a signal, using float_s32 maths.

        Take 1 new sample and return the updated envelope. Input should be
        scaled with 0dB = 1.0.

        Note this returns the mean**2 value, there is no need to do the sqrt() as
        if the output is converted to dB, 10log10() can be taken instead of
        20log10().

        """
        if isinstance(sample, utils.float_s32):
            # don't do anything if we got float_s32, this function was probably
            # called from a limiter or compressor
            sample_s32 = sample
        else:
            # if input isn't float_s32, convert it
            sample_s32 = utils.float_s32(sample)

        # for rms use power (sample**2)
        sample_mag = sample_s32 * sample_s32

        # see if we're attacking or decaying
        if sample_mag > self.envelope_s32:
            alpha = self.attack_alpha_uq30
        else:
            alpha = self.release_alpha_uq30

        # do exponential moving average
        self.envelope_s32 = utils.float_s32_ema(sample_mag, self.envelope_s32, alpha)

        # if we got floats, return floats, otherwise return float_s32
        if isinstance(sample, utils.float_s32):
            return self.envelope_s32
        else:
            return float(self.envelope_s32)


class limiter_base(dspg.dsp_block):
    # Limiter after Zolzer's DAFX & Guy McNally's "Dynamic Range Control of
    # Digital Audio Signals"
    def __init__(self, fs, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(Q_sig)

        # attack times simplified from McNally, seem pretty close
        T = 1/fs
        self.attack_alpha = 2*T / attack_t
        self.release_alpha = 2*T / release_t
        self.gain = 1

        # These are defined differently for peak and RMS limiters
        self.threshold = None
        self.env_detector = None

        self.attack_alpha_uq30 = utils.uq_2_30(round(self.attack_alpha * 2**30))
        self.release_alpha_uq30 = utils.uq_2_30(round(self.release_alpha * 2**30))
        self.threshold_s32 = None
        self.gain_s32 = utils.float_s32([2**30, -30])

    def reset_state(self):
        self.env_detector.reset_state()
        self.gain = 1
        self.gain_s32 = utils.float_s32(1)

    def process(self, sample):
        # get envelope from envelope detector
        envelope = self.env_detector.process(sample)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # if envelope below threshold, apply unity gain, otherwise scale down
        new_gain = self.threshold/envelope
        new_gain = min(1, new_gain)

        # see if we're attacking or decaying
        if new_gain < self.gain:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.gain = ((1-alpha) * self.gain) + (alpha * new_gain)

        # apply gain to input
        y = self.gain*sample
        return y, new_gain, envelope

    def process_int(self, sample):
        sample = utils.float_s32(sample, self.Q_sig)

        # get envelope from envelope detector
        envelope = self.env_detector.process_int(sample)
        # avoid /0
        if envelope.mant == 0:
            envelope.mant = 1
            envelope.exp = -60

        # if envelope below threshold, apply unity gain, otherwise scale down
        new_gain = self.threshold_s32/envelope
        new_gain = utils.float_s32_min(utils.float_s32(1), new_gain)

        # see if we're attacking or decaying
        if new_gain < self.gain_s32:
            alpha = self.attack_alpha_uq30
        else:
            alpha = self.release_alpha_uq30

        # do exponential moving average
        self.gain_s32 = utils.float_s32_ema(new_gain, self.gain_s32, alpha)

        # apply gain
        y = self.gain_s32*sample

        # quantize before return
        y = utils.float_s32_use_exp(y, -27)

        return float(y), float(new_gain), float(envelope)


class limiter_peak(limiter_base):

    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold, self.Q_sig)
        self.env_detector = envelope_detector_peak(fs, attack_t=attack_t,
                                                   release_t=release_t,
                                                   Q_sig=self.Q_sig)


class limiter_rms(limiter_base):

    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold, self.Q_sig)
        self.env_detector = envelope_detector_rms(fs, attack_t=attack_t,
                                                  release_t=release_t,
                                                  Q_sig=self.Q_sig)


class hard_limiter_peak(limiter_peak):

    def process(self, sample):
        # do peak limiting
        y = super().process(sample)

        # hard clip if above threshold
        if y > self.threshold:
            y = self.threshold
        if y < -self.threshold:
            y = -self.threshold
        return y

    # TODO process_int, super().process_int will return float though...
    def process_int(self, sample):
        raise NotImplementedError


class soft_limiter_peak(limiter_peak):
    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0,
                 nonlinear_point=0.5, Q_sig=dspg.Q_SIG):
        super().__init__(fs, threshold_db, attack_t, release_t, delay, Q_sig)
        self.nonlinear_point = nonlinear_point
        raise NotImplementedError

    # TODO soft clipping
    def process(self, sample):
        raise NotImplementedError

    def process_int(self, sample):
        raise NotImplementedError


class lookahead_limiter_peak(limiter_base):
    # peak limiter with built in delay for avoiding clipping
    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold, self.Q_sig)
        self.env_detector = envelope_detector_peak(fs, attack_t=attack_t,
                                                   release_t=release_t,
                                                   Q_sig=self.Q_sig)

        self.delay = np.ceil(attack_t*fs)
        self.delay_line = np.zeros(self.delay_line)
        raise NotImplementedError

    def process(self, sample):
        raise NotImplementedError

    def process_int(self, sample):
        raise NotImplementedError


class lookahead_limiter_rms(limiter_base):
    # rms limiter with built in delay for avoiding clipping
    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold, self.Q_sig)
        self.env_detector = envelope_detector_rms(fs, attack_t=attack_t,
                                                  release_t=release_t,
                                                  Q_sig=self.Q_sig)
        self.delay = np.ceil(attack_t*fs)
        self.delay_line = np.zeros(self.delay_line)
        raise NotImplementedError

    def process(self, sample):
        raise NotImplementedError

    def process_int(self, sample):
        raise NotImplementedError

# TODO lookahead limiters and compressors
# TODO add soft limiter
# TODO add RMS compressors
# TODO add peak compressors
# TODO add soft knee compressors
# TODO add lookup compressors w/ some magic interface


if __name__ == "__main__":
    import audio_dsp.dsp.signal_gen as gen

    fs = 48000
    x1 = np.ones(int(fs*0.5))
    x2 = 0.1*np.ones(int(fs*0.5))

    x = np.concatenate((x1, x2, x1, x2))
    # x = gen.sin(fs, 0.2, 997*4, 1)

    t = np.arange(len(x))/fs

    threshold = -6
    at = 0.01

    lt = limiter_peak(fs, threshold, at, 0.3)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

    lt.reset_state()

    y_int = np.zeros_like(x)
    f_int = np.zeros_like(x)
    env_int = np.zeros_like(x)

    import cProfile

    with cProfile.Profile() as pr:
        for n in range(len(y)):
            y_int[n], f_int[n], env_int[n] = lt.process_int(x[n])
        pr.print_stats(sort='time')


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
