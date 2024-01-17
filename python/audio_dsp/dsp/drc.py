import numpy as np
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.signal_gen as gen


class envelope_detector_peak(dspg.dsp_block):
    def __init__(self, fs, detect_t=None, attack_t=None, release_t=None, Q_sig=dspg.Q_SIG):
        super().__init__(fs, Q_sig)

        if detect_t and (attack_t or release_t):
            ValueError("either detect_t OR (attack_t AND release_t) must be specified")
        elif detect_t:
            attack_t = detect_t
            release_t = detect_t

        T = 1/fs
        self.attack_alpha = 2*T / attack_t
        self.release_alpha = 2*T / release_t
        self.envelope = 0

        self.attack_alpha_s32 = utils.float_s32(self.attack_alpha)
        self.release_alpha_s32 = utils.float_s32(self.release_alpha)
        self.envelope_s32 = utils.float_s32(0)

    def process(self, sample):
        sample_mag = abs(sample)
        if sample_mag > self.envelope:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        self.envelope = ((1-alpha) * self.envelope) + (alpha * sample_mag)

    def process_int(self, sample):
        float_flag = False
        if not isinstance(sample, utils.float_s32):
            sample = utils.float_s32(sample)
            float_flag = True

        sample_mag = abs(sample)

        if sample_mag > self.envelope_s32:
            alpha = self.attack_alpha_s32
        else:
            alpha = self.release_alpha_s32

        self.envelope_s32 = ((utils.float_s32(1)-alpha) * self.envelope_s32) + (alpha * sample_mag)

        # if we got floats, return floats, otherwise return float_s32
        if float_flag:
            return float(self.envelope_s32)
        else:
            return self.envelope_s32


class envelope_detector_rms(envelope_detector_peak):
    # note this returns the mean**2, no point doing the sqrt() as if the
    # ouptut is converted to dB, 10log10() can be taken instead of 20log10()
    def process(self, sample):
        sample_mag = sample**2
        if sample_mag > self.envelope:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        self.envelope = ((1-alpha) * self.envelope) + (alpha * sample_mag)

        return self.envelope


class limiter_base(dspg.dsp_block):
    # Limiter after Zolzer's DAFX & Guy McNally's "Dynamic Range Control of
    # Digital Audio Signals"
    def __init__(self, fs, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(Q_sig)

        T = 1/fs
        self.attack_alpha = 2*T / attack_t
        self.release_alpha = 2*T / release_t
        self.gain = 1

        # These are defined differently for peak and RMS limiters
        self.threshold = None
        self.env_detector = None

        self.attack_alpha_s32 = utils.float_s32(self.attack_alpha)
        self.release_alpha_s32 = utils.float_s32(self.release_alpha)
        self.threshold_s32 = None
        self.gain_s32 = utils.float_s32(1)

    def process(self, sample):

        envelope = self.env_detector.process(sample)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        new_gain = self.threshold/envelope
        new_gain = min(1, new_gain)

        if new_gain < self.gain:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        self.gain = ((utils.float_s32(1)-alpha) * self.gain) + (alpha * new_gain)

        y = self.gain*sample
        return y, new_gain, envelope

    def process_int(self, sample):
        sample = utils.float_s32(sample)

        envelope = self.env_detector.process_int(sample)
        # avoid /0
        if envelope.mant == 0:
            envelope.mant = 1

        new_gain = self.threshold_s32/envelope
        new_gain = utils.float_s32_min(utils.float_s32(1), new_gain)

        if new_gain < self.gain_s32:
            alpha = self.attack_alpha_s32
        else:
            alpha = self.release_alpha_s32

        self.gain_s32 = ((utils.float_s32(1)-alpha) * self.gain_s32) + (alpha * new_gain)

        y = self.gain_s32*sample

        return float(y), float(new_gain), float(envelope)


class limiter_peak(limiter_base):

    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold)
        self.env_detector = envelope_detector_peak(fs, attack_t=attack_t, release_t=release_t)


class limiter_rms(limiter_base):

    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_db)
        self.threshold_s32 = utils.float_s32(self.threshold)
        self.env_detector = envelope_detector_rms(fs, attack_t=attack_t, release_t=release_t)


class hard_limiter_peak(limiter_peak):

    def process(self, sample):
        y = super().process(sample)
        if y > self.threshold:
            y = self.threshold
        if y < -self.threshold:
            y = -self.threshold
        return y


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
