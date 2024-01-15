import numpy as np
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.signal_gen as gen


class envelope_detector_peak(dspg.dsp_block):
    def __init__(self, fs, detect_t=None, attack_t=None, release_t=None, Q_sig=dspg.Q_SIG):
        super().__init__(Q_sig)

        if detect_t and (attack_t or release_t):
            ValueError("either detect_t OR (attack_t AND release_t) must be specified")
        elif detect_t:
            attack_t = detect_t
            release_t = detect_t

        T = 1/fs
        self.attack_alpha = 2*T / attack_t
        self.release_alpha = 2*T / release_t
        self.envelope = 0

    def process(self, sample):
        sample_mag = abs(sample)
        if sample_mag > self.envelope:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        self.envelope = ((1-alpha) * self.envelope) + (alpha * sample_mag)

        return self.envelope


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

        self.gain = ((1-alpha) * self.gain) + (alpha * new_gain)

        y = self.gain*sample
        return y, new_gain, envelope


class limiter_peak(limiter_base):

    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db2gain(threshold_db)
        self.env_detector = envelope_detector_peak(fs, attack_t=attack_t, release_t=release_t)


class limiter_rms(limiter_base):

    def __init__(self, fs, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, attack_t, release_t, delay, Q_sig)

        self.threshold = utils.db_pow2gain(threshold_db)
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


if __name__ == "__main__":
    fs = 48000
    x1 = np.ones(fs*2)
    x2 = 0.02*np.ones(fs*2)

    x = np.concatenate((x1, x2, x1, x2))
    # x = gen.sin(fs, 2, 997*4, 1)

    t = np.arange(len(x))/fs

    threshold = -20
    at = 0.01

    lt = limiter_peak(fs, threshold, at, 0.3)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

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
