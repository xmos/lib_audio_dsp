import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt

from . import utils as utils


class biquad():
    def __init__(self, coeffs: list, b_shift=0, Q_sig=30):

        self.b_shift = b_shift
        self.Q_sig = Q_sig

        # coeffs should be in the form [b0 b1 b2 -a1 -a2], and normalized by a0
        assert len(coeffs) == 5, "coeffs should be in the form [b0 b1 b2 -a1 -a2]"
        self.coeffs, self.int_coeffs = round_and_check(coeffs, self.b_shift)

        # state variables
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0

        return

    def update_coeffs(self, new_coeffs: list):
        new_coeffs = round_and_check(new_coeffs, self.b_shift)
        self.coeffs = new_coeffs

        return

    def process(self, sample):
        # process a single sample using direct form 1
        y = (self.coeffs[0]*sample +
             self.coeffs[1]*self.x1 +
             self.coeffs[2]*self.x2 +
             self.coeffs[3]*self.y1 +
             self.coeffs[4]*self.y2)

        self.x2 = self.x1
        self.x1 = sample
        self.y2 = self.y1
        self.y1 = y

        y = y * 2**-self.b_shift

        return y

    def process_int(self, sample):

        sample_int = np.round(sample * 2**self.Q_sig).astype(np.int32)

        # process a single sample using direct form 1
        y = ((self.int_coeffs[0].astype(np.int64)*sample_int) +
             (self.int_coeffs[1].astype(np.int64)*self.x1) +
             (self.int_coeffs[2].astype(np.int64)*self.x2) +
             (self.int_coeffs[3].astype(np.int64)*self.y1) +
             (self.int_coeffs[4].astype(np.int64)*self.y2))

        # in an ideal world, we do (y << -self.b_shift) here, but the rest of
        # the VPU must come first 

        # rounding back to int_32 VPU style
        y = y + 2**29
        y = (y >> 30).astype(np.int32)

        # save states
        self.x2 = np.array(self.x1).astype(np.int32)
        self.x1 = np.array(sample_int).astype(np.int32)
        self.y2 = np.array(self.y1).astype(np.int32)
        self.y1 = np.array(y).astype(np.int32)

        # compensate for coefficients
        y = (y << -self.b_shift).astype(np.int32)

        y_flt = (y.astype(np.double)*2**-self.Q_sig).astype(np.double)

        return y_flt

    def freq_response(self, nfft=512):
        b = [self.coeffs[0], self.coeffs[1], self.coeffs[2]]
        b = apply_biquad_bshift(b, -self.b_shift)
        a = [1, -self.coeffs[3], -self.coeffs[4]]
        w, h = spsig.freqz(b, a, worN=nfft)

        return w, h


def biquad_lowpass(fs, f, q):
    coeffs = make_biquad_lowpass(fs, f, q)
    return biquad(coeffs)


def biquad_highpass(fs, f, q):
    coeffs = make_biquad_highpass(fs, f, q)
    return biquad(coeffs)


def biquad_bandpass(fs, f, bw):
    # bw is bandwidth in octaves
    coeffs = make_biquad_bandpass(fs, f, bw)
    return biquad(coeffs)


def biquad_bandstop(fs, f, bw):
    # bw is bandwidth in octaves
    coeffs = make_biquad_bandstop(fs, f, bw)
    return biquad(coeffs)


def biquad_notch(fs, f, q):
    coeffs = make_biquad_notch(fs, f, q)
    return biquad(coeffs)


def biquad_allpass(fs, f, q):
    coeffs = make_biquad_allpass(fs, f, q)
    return biquad(coeffs)


def biquad_peaking(fs, f, q, boost_db):
    coeffs = make_biquad_peaking(fs, f, q, boost_db)
    return biquad(coeffs)


def biquad_constant_q(fs, f, q, boost_db):
    coeffs = make_biquad_constant_q(fs, f, q, boost_db)
    return biquad(coeffs)


def biquad_lowshelf(fs, f, q, boost_db):
    # q is similar to standard low pass, i.e. > 0.707 will yield peakiness
    # the level change at f will be boost_db/2
    coeffs = make_biquad_lowshelf(fs, f, q, boost_db)
    return biquad(coeffs)


def biquad_highshelf(fs, f, q, boost_db):
    # q is similar to standard high pass, i.e. > 0.707 will yield peakiness
    # the level change at f will be boost_db/2
    coeffs = make_biquad_highshelf(fs, f, q, boost_db)
    return biquad(coeffs)


def biquad_linkwitz(fs, f0, q0, fp, qp):
    # used for changing one low frequency roll off slope for another,
    # e.g. in a loudspeaker
    coeffs = make_biquad_linkwitz(fs, f0, q0, fp, qp)
    return biquad(coeffs)


def round_to_q30(coeffs, b_shift):
    rounded_coeffs = [None] * len(coeffs)
    int_coeffs = [None] * len(coeffs)

    Q = 30 + b_shift
    for n in range(len(coeffs)):
        # scale to Q30 ints
        rounded_coeffs[n] = np.round(coeffs[n] * 2**Q)
        # check for overflow
        assert (rounded_coeffs[n] > -2**31 and rounded_coeffs[n] < (2**31 - 1)), \
            "Filter coefficient will overflow (%.4f, %d), reduce gain" % (coeffs[n], n)

        int_coeffs[n] = rounded_coeffs[n].astype(np.int32)
        # rescale to floats
        rounded_coeffs[n] = rounded_coeffs[n]/2**Q

    return rounded_coeffs, int_coeffs


def apply_biquad_gain(coeffs, gain_db):
    # apply linear gain to the b coefficients
    gain = 10 ** (gain_db/20)
    coeffs[0] = coeffs[0] * gain
    coeffs[1] = coeffs[1] * gain
    coeffs[2] = coeffs[2] * gain

    return coeffs


def apply_biquad_bshift(coeffs, b_shift):
    # apply linear bitshift to the b coefficients
    gain = 2**b_shift
    coeffs[0] = coeffs[0] * gain
    coeffs[1] = coeffs[1] * gain
    coeffs[2] = coeffs[2] * gain

    return coeffs


def normalise_biquad(coeffs):
    # divide by a0, make a1 and a2 negative
    coeffs = [coeffs[0]/coeffs[3],
              coeffs[1]/coeffs[3],
              coeffs[2]/coeffs[3],
              -coeffs[4]/coeffs[3],
              -coeffs[5]/coeffs[3]]

    return coeffs


def round_and_check(coeffs, b_shift=0):
    # round to int32 precision
    coeffs, int_coeffs = round_to_q30(coeffs, b_shift)

    # check filter is stable
    poles = np.roots([1, -coeffs[3], -coeffs[4]])
    assert np.all(np.abs(poles) < 1), "Poles lie outside the unit circle, the filter is unstable"

    return coeffs, int_coeffs


def make_biquad_bypass(fs):
    # take in fs to match other apis
    coeffs = [1, 0, 0, 0, 0]

    return coeffs


def make_biquad_mute(fs):
    # take in fs to match other apis
    coeffs = [0, 0, 0, 0, 0]

    return coeffs


def make_biquad_lowpass(fs, filter_freq, q_factor):

    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2 * q_factor)

    b0 = (+1.0 - np.cos(w0)) / 2.0
    b1 = (+1.0 - np.cos(w0))
    b2 = (+1.0 - np.cos(w0)) / 2.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_highpass(fs, filter_freq, q_factor):

    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2 * q_factor)

    b0 = (1.0 + np.cos(w0)) / 2.0
    b1 = -(1.0 + np.cos(w0))
    b2 = (1.0 + np.cos(w0)) / 2.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


# Constant 0 dB peak gain
def make_biquad_bandpass(fs, filter_freq, BW):

    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0) * np.sinh(np.log(2)/2 * BW * w0/np.sin(w0))

    b0 = alpha
    b1 = +0.0
    b2 = -alpha
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


# Constant 0 dB peak gain
def make_biquad_bandstop(fs, filter_freq, BW):

    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0) * np.sinh(np.log(2)/2 * BW * w0/np.sin(w0))

    b0 = +1.0
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_notch(fs, filter_freq, q_factor):

    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2.0 * q_factor)

    b0 = +1.0
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_allpass(fs, filter_freq, q_factor):

    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2.0 * q_factor)

    b0 = +1.0 - alpha
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0 + alpha
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_peaking(fs, filter_freq, q_factor, boost_db):

    A = np.sqrt(10 ** (boost_db/20))
    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2.0 * q_factor)

    b0 = +1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0 - alpha * A
    a0 = +1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha / A

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_constant_q(fs, filter_freq, q_factor, boost_db):

    # https://www.musicdsp.org/en/latest/Filters/37-zoelzer-biquad-filters.html

    V = 10 ** (boost_db/20)
    w0 = 2.0 * np.pi * filter_freq/fs
    K = np.tan(w0/2)

    if boost_db > 0:
        b0 = 1 + V*K/q_factor + K**2
        b1 = 2*(K**2 - 1)
        b2 = 1 - V*K/q_factor + K**2
        a0 = 1 + K/q_factor + K**2
        a1 = 2*(K**2 - 1)
        a2 = 1 - K/q_factor + K**2
    else:
        V = 1/V
        b0 = 1 + K/q_factor + K**2
        b1 = 2*(K**2 - 1)
        b2 = 1 - K/q_factor + K**2
        a0 = 1 + V*K/q_factor + K**2
        a1 = 2*(K**2 - 1)
        a2 = 1 - V*K/q_factor + K**2

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_lowshelf(fs, filter_freq, q_factor, gain_db):

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2*q_factor)

    b0 = A*((A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b1 = 2*A*((A-1) - (A+1)*np.cos(w0))
    b2 = A*((A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a0 = (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a1 = -2*((A-1) + (A+1)*np.cos(w0))
    a2 = (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_highshelf(fs, filter_freq, q_factor, gain_db):

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * filter_freq/fs
    alpha = np.sin(w0)/(2*q_factor)

    b0 = A*((A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*np.cos(w0))
    b2 = A*((A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a0 = (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a1 = 2*((A-1) - (A+1)*np.cos(w0))
    a2 = (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs


def make_biquad_linkwitz(fs, f0, q0, fp, qp):

    # https://www.linkwitzlab.com/filters.htm#9
    # https://www.minidsp.com/applications/advanced-tools/linkwitz-transform

    fc = (f0 + fp) / 2
    fc = fc
    low = 1
    # 1 = 1

    d0i = (2*np.pi * f0)**2
    d1i = (2*np.pi * f0)/q0
    d2i = 1

    c0i = (2*np.pi * fp)**2
    c1i = (2*np.pi * fp)/qp
    c2i = 1

    gn = (2*np.pi*fc)/(np.tan(np.pi*fc/fs))
    cci = c0i+gn*c1i+(gn**2)

    a0 = cci
    a1 = 2*(c0i-(gn**2))
    a2 = (c0i-gn*c1i+(gn**2))

    b0 = (d0i+gn*d1i+(gn**2))
    b1 = 2*(d0i-(gn**2))
    b2 = (d0i-gn*d1i+(gn**2))

    coeffs = (b0, b1, b2, a0, a1, a2)
    coeffs = normalise_biquad(coeffs)

    return coeffs



if __name__ == "__main__":

    fs = 48000

    biquad_1 = biquad(make_biquad_notch(fs, 20, 1), 0, Q_sig=30)
    biquad_2 = biquad(make_biquad_notch(fs, 20, 1), -3, Q_sig=30)
    biquad_3 = biquad(make_biquad_notch(fs, 20, 1), 0, Q_sig=27)
    biquad_4 = biquad(make_biquad_notch(fs, 20, 1), -3, Q_sig=27)
    biquad_5 = biquad(make_biquad_notch(fs, 20, 1), -3, Q_sig=27)

    t = np.arange(fs*4)/fs
    # signal = spsig.chirp(t, 20, 1, 20000, 'log', phi=-90)
    signal = np.sin(2*np.pi*997*t)

    output_1 = np.zeros(len(signal))
    output_2 = np.zeros(len(signal))
    output_3 = np.zeros(len(signal))
    output_4 = np.zeros(len(signal))
    output_5 = np.zeros(len(signal))

    for n in np.arange(len(signal)):
        output_1[n] = biquad_1.process_int(signal[n])
        output_2[n] = biquad_2.process_int(signal[n])
        output_3[n] = biquad_3.process_int(signal[n])
        output_4[n] = biquad_4.process_int(signal[n])
        output_5[n] = biquad_5.process(signal[n])

    # plt.plot(signal)
    # plt.plot(output_1)
    # plt.plot(output_2)

    plt.psd(output_1, 1024*16, fs, window=spsig.windows.blackmanharris(1024*16))
    plt.psd(output_2, 1024*16, fs, window=spsig.windows.blackmanharris(1024*16))
    plt.psd(output_3, 1024*16, fs, window=spsig.windows.blackmanharris(1024*16))
    plt.psd(output_4, 1024*16, fs, window=spsig.windows.blackmanharris(1024*16))
    plt.psd(output_5, 1024*16, fs, window=spsig.windows.blackmanharris(1024*16))

    ax = plt.gca()
    ax.set_xscale('log')
    plt.legend(["Q30", "Q30, b_shift -3", "Q27", "Q27, b_shift -3", "double"])
    plt.show()

    pass
    exit()

    # fun linkwitz transform test
    biquad_1 = biquad_linkwitz(fs, 100, 1.21, 80, 0.5)
    biquad_2 = biquad_highpass(fs, 100, 1.21)
    biquad_3 = biquad_highpass(fs, 80, 0.5)


    t = np.arange(fs)/fs
    signal = spsig.chirp(t, 20, 1, 20000, 'log', phi=-90)

    output_1 = np.zeros(fs)
    output_2 = np.zeros(fs)

    # for n in np.arange(fs):
    #     output_1[n] = biquad_1.process(signal[n])
    #     output_2[n] = biquad_2.process(signal[n])

    # plt.plot(signal)
    # plt.plot(output_2)
    # plt.plot(output_1)

    w, h1 = biquad_1.freq_response(2048)
    w, h2 = biquad_2.freq_response(2048)
    w, h3 = biquad_3.freq_response(2048)


    fig, figs = plt.subplots(2, 1)

    figs[0].semilogx(w/(2*np.pi)*fs, utils.db(h1*h2))
    figs[0].semilogx(w/(2*np.pi)*fs, utils.db(h3))
    figs[0].grid()

    figs[1].semilogx(w/(2*np.pi)*fs, np.angle(h1*h2))
    figs[1].semilogx(w/(2*np.pi)*fs, np.angle(h3))
    figs[1].grid()

    plt.show()

    np.testing.assert_allclose(utils.db(h1*h2), utils.db(h3), rtol=0, atol=1e-04)

    pass
