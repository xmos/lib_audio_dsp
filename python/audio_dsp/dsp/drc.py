from copy import deepcopy

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
    attack_alpha_f32 : np.float32
        attack_alpha in 32-biti float format.
    release_alpha_f32 : np.float32
        release_alpha in 32-bit float format.
    envelope_f32 : np.float32
        current envelope value in 32-bit float format.

    """

    def __init__(self, fs, n_chans=1, attack_t=None, release_t=None, detect_t=None, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

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

        # very long times might quantize to zero, maybe just limit a better way
        assert self.attack_alpha > 0
        assert self.release_alpha > 0

        self.attack_alpha_f32 = np.float32(self.attack_alpha)
        self.release_alpha_f32 = np.float32(self.release_alpha)

        # very long times might quantize to zero
        assert self.attack_alpha_f32 > np.float32(0)
        assert self.attack_alpha_f32 > np.float32(0)

        # initalise envelope state
        self.reset_state()

    def reset_state(self):
        """Reset the envelope to zero."""
        self.envelope = [0] * self.n_chans
        self.envelope_f32 = [np.float32(0)] * self.n_chans

    def process(self, sample, channel=0):
        """
        Update the peak envelope for a signal, using floating point maths.

        Take 1 new sample and return the updated envelope. Input should be
        scaled with 0dB = 1.0.

        """
        sample_mag = abs(sample)

        # see if we're attacking or decaying
        if sample_mag > self.envelope[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1-alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_f32(self, sample, channel=0):
        """
        Update the peak envelope for a signal, using np.float32 maths.

        Take 1 new sample and return the updated envelope. If the input is
        np.float32, return a np.float32, otherwise expect float input and return
        float output.

        """
        if isinstance(sample, np.float32):
            # don't do anything if we got np.float32, this function was probably
            # called from a limiter or compressor
            sample_f32 = sample
        else:
            # if input isn't np.float32, convert it
            sample_f32 = np.float32(sample)

        sample_mag = abs(sample_f32)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_f32[channel]:
            alpha = self.attack_alpha_f32
        else:
            alpha = self.release_alpha_f32

        # do exponential moving average
        self.envelope_f32[channel] = self.envelope_f32[channel] + alpha * (sample_mag - self.envelope_f32[channel])

        # if we got floats, return floats, otherwise return np.float32
        if isinstance(sample, np.float32):
            return self.envelope_f32[channel]
        else:
            return float(self.envelope_f32[channel])
        
    def process_frame_f32(self, frame):
        # same as generic, but only take 1st output
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_f32(this_chan[sample],
                                                 channel=chan)

        return output


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
    attack_alpha_uq30 : np.float32
        attack_alpha in 32-bit float format.
    release_alpha_uq30 : np.float32
        release_alpha in 32-bit float format.
    envelope_s32 : np.float32
        current mean squared envelope value in 32-bit float format.

    """

    def process(self, sample, channel=0):
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
        if sample_mag > self.envelope[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1-alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_f32(self, sample, channel=0):
        """
        Update the RMS envelope for a signal, using np.float32 maths.

        Take 1 new sample and return the updated envelope. Input should be
        scaled with 0dB = 1.0.

        Note this returns the mean**2 value, there is no need to do the sqrt() as
        if the output is converted to dB, 10log10() can be taken instead of
        20log10().

        """
        if isinstance(sample, np.float32):
            # don't do anything if we got np.float32, this function was probably
            # called from a limiter or compressor
            sample_f32 = sample
        else:
            # if input isn't np.float32, convert it
            sample_f32 = np.float32(sample)

        # for rms use power (sample**2)
        sample_mag = sample_f32 * sample_f32

        # see if we're attacking or decaying
        if sample_mag > self.envelope_f32[channel]:
            alpha = self.attack_alpha_f32
        else:
            alpha = self.release_alpha_f32

        # do exponential moving average
        self.envelope_f32[channel] = self.envelope_f32[channel] + alpha * (sample_mag - self.envelope_f32[channel])

        # if we got floats, return floats, otherwise return np.float32
        if isinstance(sample, np.float32):
            return self.envelope_f32[channel]
        else:
            return float(self.envelope_f32[channel])


class limiter_base(dspg.dsp_block):
    # Limiter after Zolzer's DAFX & Guy McNally's "Dynamic Range Control of
    # Digital Audio Signals"
    def __init__(self, fs, n_chans, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        # attack times simplified from McNally, seem pretty close
        T = 1/fs
        self.attack_alpha = 2*T / attack_t
        self.release_alpha = 2*T / release_t
        self.gain = [1] * n_chans

        # These are defined differently for peak and RMS limiters
        self.threshold = None
        self.env_detector = None

        self.attack_alpha_f32 = np.float32(self.attack_alpha)
        self.release_alpha_f32 = np.float32(self.release_alpha)
        self.threshold_f32 = None
        self.gain_f32 = [np.float32(1)] * n_chans

    def reset_state(self):
        self.env_detector.reset_state()
        self.gain = [1] * self.n_chans
        self.gain_f32 = [np.float32(1)] * self.n_chans

    def process(self, sample, channel=0):
        # get envelope from envelope detector
        envelope = self.env_detector.process(sample, channel)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # if envelope below threshold, apply unity gain, otherwise scale down
        new_gain = self.threshold/envelope
        new_gain = min(1, new_gain)

        # see if we're attacking or decaying
        if new_gain < self.gain[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.gain[channel] = ((1-alpha) * self.gain[channel]) + (alpha * new_gain)

        # apply gain to input
        y = self.gain[channel]*sample
        return y, new_gain, envelope

    def process_f32(self, sample, channel=0):
        # quantize
        sample = utils.float_s32(sample)
        sample = utils.float_s32_use_exp(sample, -27)
        sample = np.float32(float(sample))

        # get envelope from envelope detector
        envelope = self.env_detector.process_f32(sample, channel)
        # avoid /0
        if envelope == np.float32(0):
            envelope = np.float32(1e-20)

        # if envelope below threshold, apply unity gain, otherwise scale down
        new_gain = self.threshold_f32/envelope
        new_gain = new_gain if new_gain < np.float32(1) else np.float32(1)

        # see if we're attacking or decaying
        if new_gain < self.gain_f32[channel]:
            alpha = self.attack_alpha_f32
        else:
            alpha = self.release_alpha_f32

        # do exponential moving average
        self.gain_f32[channel] = self.gain_f32[channel] + alpha * (new_gain - self.gain_f32[channel])

        # apply gain
        y = self.gain_f32[channel]*sample

        # quantize before return
        y = utils.float_s32(y)
        y = utils.float_s32_use_exp(y, -27)

        return np.float32(float(y)), float(new_gain), float(envelope)

    def process_frame(self, frame):
        # same as generic, but only take 1st output
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process(this_chan[sample],
                                                 channel=chan)[0]

        return output

    def process_frame_f32(self, frame):
        # same as generic, but only take 1st output
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_f32(this_chan[sample],
                                                 channel=chan)[0]

        return output


class limiter_peak(limiter_base):

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_f32 = np.float32(self.threshold)
        self.env_detector = envelope_detector_peak(fs, n_chans=n_chans,
                                                   attack_t=attack_t,
                                                   release_t=release_t,
                                                   Q_sig=self.Q_sig)


class limiter_rms(limiter_base):

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_db)
        self.threshold_f32 = np.float32(self.threshold)
        self.env_detector = envelope_detector_rms(fs, n_chans=n_chans,
                                                  attack_t=attack_t,
                                                  release_t=release_t,
                                                  Q_sig=self.Q_sig)


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
        self.threshold_f32 = np.float32(self.threshold)
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
        self.threshold_f32 = np.float32(self.threshold)
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
