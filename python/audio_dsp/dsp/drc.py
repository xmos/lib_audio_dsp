import numpy as np
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.signal_gen as gen


class envelope_detector_peak(dspg.dsp_block):
    def __init__(self, fs, n_chans=1, attack_t=None, release_t=None, detect_t=None, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        if detect_t and (attack_t or release_t):
            ValueError("either detect_t OR (attack_t AND release_t) must be specified")
        elif detect_t:
            attack_t = detect_t
            release_t = detect_t

        # attack times simplified from McNally, seem pretty close
        T = 1/fs
        self.attack_alpha = 2*T / attack_t
        self.release_alpha = 2*T / release_t

        self.attack_alpha_uq30 = utils.uq_2_30(round(self.attack_alpha * 2**30))
        self.release_alpha_uq30 = utils.uq_2_30(round(self.release_alpha * 2**30))

        # initalise envelope state
        self.reset_state()

    def reset_state(self):
        self.envelope = [0] * self.n_chans
        self.envelope_s32 = [utils.float_s32([0, -self.Q_sig])] * self.n_chans

    def process(self, sample, channel=0):
        sample_mag = abs(sample)

        # see if we're attacking or decaying
        if sample_mag > self.envelope:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1-alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_int(self, sample, channel=0):
        if isinstance(sample, utils.float_s32):
            sample_s32 = sample
        else:
            sample_s32 = utils.float_s32(sample)

        sample_mag = abs(sample_s32)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_s32[channel]:
            alpha = self.attack_alpha_uq30
        else:
            alpha = self.release_alpha_uq30

        # do exponential moving average
        self.envelope_s32[channel] = utils.float_s32_ema(sample_mag, self.envelope_s32[channel], alpha)

        # if we got floats, return floats, otherwise return float_s32
        if isinstance(sample, utils.float_s32):
            return self.envelope_s32[channel]
        else:
            return float(self.envelope_s32[channel])


class envelope_detector_rms(envelope_detector_peak):
    # note this returns the mean**2, no point doing the sqrt() as if the
    # ouptut is converted to dB, 10log10() can be taken instead of 20log10()
    def process(self, sample, channel=0):
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

    def process_int(self, sample, channel=0):
        if isinstance(sample, utils.float_s32):
            sample_s32 = sample
        else:
            sample_s32 = utils.float_s32(sample)

        # for rms use power (sample**2)
        sample_mag = sample_s32 * sample_s32

        # see if we're attacking or decaying
        if sample_mag > self.envelope_s32[channel]:
            alpha = self.attack_alpha_uq30
        else:
            alpha = self.release_alpha_uq30

        # do exponential moving average
        self.envelope_s32[channel] = utils.float_s32_ema(sample_mag, self.envelope_s32[channel], alpha)

        # if we got floats, return floats, otherwise return float_s32
        if isinstance(sample, utils.float_s32):
            return self.envelope_s32[channel]
        else:
            return float(self.envelope_s32[channel])


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

        self.attack_alpha_uq30 = utils.uq_2_30(round(self.attack_alpha * 2**30))
        self.release_alpha_uq30 = utils.uq_2_30(round(self.release_alpha * 2**30))
        self.threshold_s32 = None
        self.gain_s32 = [utils.float_s32([2**30, -30])] * n_chans

    def reset_state(self):
        self.env_detector.reset_state()
        self.gain = [1] * self.n_chans
        self.gain_s32 = [utils.float_s32([2**30, -30])] * self.n_chans

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

    def process_int(self, sample, channel=0):
        sample = utils.float_s32(sample, self.Q_sig)

        # get envelope from envelope detector
        envelope = self.env_detector.process_int(sample, channel)
        # avoid /0
        if envelope.mant == 0:
            envelope.mant = 1
            envelope.exp = -60

        # if envelope below threshold, apply unity gain, otherwise scale down
        new_gain = self.threshold_s32/envelope
        new_gain = utils.float_s32_min(utils.float_s32(1), new_gain)

        # see if we're attacking or decaying
        if new_gain < self.gain_s32[channel]:
            alpha = self.attack_alpha_uq30
        else:
            alpha = self.release_alpha_uq30

        # do exponential moving average
        self.gain_s32[channel] = utils.float_s32_ema(new_gain, self.gain_s32[channel], alpha)

        # apply gain
        y = self.gain_s32[channel]*sample

        # quantize before return
        y = utils.float_s32_use_exp(y, -27)

        return float(y), float(new_gain), float(envelope)


class limiter_peak(limiter_base):

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold, self.Q_sig)
        self.env_detector = envelope_detector_peak(fs, n_chans=n_chans,
                                                   attack_t=attack_t,
                                                   release_t=release_t,
                                                   Q_sig=self.Q_sig)


class limiter_rms(limiter_base):

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold, self.Q_sig)
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


# TODO add soft limiter
# TODO add RMS compressors
# TODO add peak compressors
# TODO add soft knee compressors
# TODO add lookup compressors w/ some magic interface


if __name__ == "__main__":
    fs = 48000
    x1 = np.ones(fs*2)
    x2 = 0.02*np.ones(fs*2)

    x = np.concatenate((x1, x2, x1, x2))
    # x = gen.sin(fs, 0.2, 997*4, 1)

    t = np.arange(len(x))/fs

    threshold = -20
    at = 0.01

    lt = limiter_peak(fs, threshold, at, 0.3)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process_int(x[n])

    thresh_passed = np.argmax(utils.db(env) > threshold)
    sig_3dB = np.argmax(utils.db(y) < (threshold + 3))

    measured_at = t[sig_3dB] - t[thresh_passed]
    print(measured_at)

    import matplotlib.pyplot as plt
    plt.plot(t, utils.db(x))
    plt.plot(t, utils.db(y))
    plt.plot(t, utils.db(env))
    plt.plot(t, utils.db(f))

    plt.legend(["x", "y", "env", "gain"])
    plt.grid()
    plt.show()

    pass
