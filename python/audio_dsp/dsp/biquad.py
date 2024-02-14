import warnings
from copy import deepcopy

import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg

BOOST_BSHIFT = 2  # limit boosts to 12 dB gain


class biquad(dspg.dsp_block):
    def __init__(self, coeffs: list, fs, n_chans=1, b_shift=0, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.b_shift = b_shift

        # coeffs should be in the form [b0 b1 b2 -a1 -a2], and normalized by a0
        assert len(coeffs) == 5, "coeffs should be in the form [b0 b1 b2 -a1 -a2]"
        self.coeffs, self.int_coeffs = round_and_check(coeffs, self.b_shift)

        self.check_gain()

        # state variables
        self.x1 = [0]*n_chans
        self.x2 = [0]*n_chans
        self.y1 = [0]*n_chans
        self.y2 = [0]*n_chans

        return

    def update_coeffs(self, new_coeffs: list):
        new_coeffs = round_and_check(new_coeffs, self.b_shift)
        self.coeffs = new_coeffs

        return

    def process(self, sample, channel=0):
        # process a single sample using direct form 1
        y = (self.coeffs[0]*sample +
             self.coeffs[1]*self.x1[channel] +
             self.coeffs[2]*self.x2[channel] +
             self.coeffs[3]*self.y1[channel] +
             self.coeffs[4]*self.y2[channel])

        self.x2[channel] = self.x1[channel]
        self.x1[channel] = sample
        self.y2[channel] = self.y1[channel]
        self.y1[channel] = y

        y = y * 2**self.b_shift

        return y

    def process_int(self, sample, channel=0):

        sample_int = utils.int32(round(sample * 2**self.Q_sig))

        # process a single sample using direct form 1
        y = utils.int64(((sample_int*self.int_coeffs[0])) +
                        ((self.x1[channel]*self.int_coeffs[1])) +
                        ((self.x2[channel]*self.int_coeffs[2])) +
                        (((self.y1[channel]*self.int_coeffs[3]) >> self.b_shift)) +
                        (((self.y2[channel]*self.int_coeffs[4]) >> self.b_shift)))

        # combine the b_shift with the >> 30
        y = y + 2**(29 - self.b_shift)
        y = utils.int32(y >> (30 - self.b_shift))

        # save states
        self.x2[channel] = utils.int32(self.x1[channel])
        self.x1[channel] = utils.int32(sample_int)
        self.y2[channel] = utils.int32(self.y1[channel])
        self.y1[channel] = utils.int32(y)

        y_flt = (float(y)*2**-self.Q_sig)

        return y_flt

    def process_vpu(self, sample, channel=0):

        sample_int = utils.int32(round(sample * 2**self.Q_sig))

        # process a single sample using direct form 1. In the VPU the ``>> 30``
        # comes before accumulation
        y = utils.vlmaccr([sample_int, self.x1[channel], self.x2[channel],
                           self.y1[channel], self.y2[channel]],
                          self.int_coeffs)

        # save states
        self.x2[channel] = utils.int32(self.x1[channel])
        self.x1[channel] = utils.int32(sample_int)
        self.y2[channel] = utils.int32(self.y1[channel])
        self.y1[channel] = utils.int32(y)

        # compensate for coefficients
        y = utils.int32(y << self.b_shift)

        y_flt = (float(y)*2**-self.Q_sig)

        return y_flt

    def process_frame_vpu(self, frame):
        # simple multichannel, but integer. Assumes no channel unique states!
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_vpu(this_chan[sample],
                                                     channel=chan)

        return output

    def freq_response(self, nfft=512):
        b = [self.coeffs[0], self.coeffs[1], self.coeffs[2]]
        b = apply_biquad_bshift(b, self.b_shift)
        a = [1, -self.coeffs[3], -self.coeffs[4]]
        f, h = spsig.freqz(b, a, worN=nfft, fs=self.fs)

        return f, h

    def check_gain(self):
        _, h = self.freq_response()
        max_gain = np.max(utils.db(h))
        if max_gain > dspg.HEADROOM_DB:
            warnings.warn("biquad gain (%.1f dB) is > headroom" % (max_gain) +
                          " (%.0f dB), overflow may occur" % dspg.HEADROOM_DB +
                          " unless signal level has previously been reduced")
        return

    def reset_state(self):
        for chan in range(self.n_chans):
            self.x1[chan] = 0
            self.x2[chan] = 0
            self.y1[chan] = 0
            self.y2[chan] = 0

        return


def biquad_bypass(fs, n_chans):
    coeffs = make_biquad_bypass(fs)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_gain(fs, n_chans, gain_db):
    coeffs = make_biquad_gain(fs, gain_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT)


def biquad_lowpass(fs, n_chans, f, q):
    coeffs = make_biquad_lowpass(fs, f, q)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_highpass(fs, n_chans, f, q):
    coeffs = make_biquad_highpass(fs, f, q)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_bandpass(fs, n_chans, f, bw):
    # bw is bandwidth in octaves
    coeffs = make_biquad_bandpass(fs, f, bw)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_bandstop(fs, n_chans, f, bw):
    # bw is bandwidth in octaves
    coeffs = make_biquad_bandstop(fs, f, bw)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_notch(fs, n_chans, f, q):
    coeffs = make_biquad_notch(fs, f, q)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_allpass(fs, n_chans, f, q):
    coeffs = make_biquad_allpass(fs, f, q)
    return biquad(coeffs, fs, n_chans=n_chans)


def biquad_peaking(fs, n_chans, f, q, boost_db):
    coeffs = make_biquad_peaking(fs, f, q, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT)


def biquad_constant_q(fs, n_chans, f, q, boost_db):
    coeffs = make_biquad_constant_q(fs, f, q, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT)


def biquad_lowshelf(fs, n_chans, f, q, boost_db):
    # q is similar to standard low pass, i.e. > 0.707 will yield peakiness
    # the level change at f will be boost_db/2
    coeffs = make_biquad_lowshelf(fs, f, q, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT)


def biquad_highshelf(fs, n_chans, f, q, boost_db):
    # q is similar to standard high pass, i.e. > 0.707 will yield peakiness
    # the level change at f will be boost_db/2
    coeffs = make_biquad_highshelf(fs, f, q, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT)


def biquad_linkwitz(fs, n_chans, f0, q0, fp, qp):
    # used for changing one low frequency roll off slope for another,
    # e.g. in a loudspeaker
    coeffs = make_biquad_linkwitz(fs, f0, q0, fp, qp)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=0)


def round_to_q30(coeffs, b_shift):
    rounded_coeffs = [None] * len(coeffs)
    int_coeffs = [None] * len(coeffs)

    Q = 30  # - b_shift
    for n in range(len(coeffs)):
        # scale to Q30 ints
        rounded_coeffs[n] = round(coeffs[n] * 2**Q)
        # check for overflow
        assert (rounded_coeffs[n] > -2**31 and rounded_coeffs[n] < (2**31 - 1)), \
            "Filter coefficient will overflow (%.4f, %d), reduce gain" % (coeffs[n], n)

        int_coeffs[n] = utils.int32(rounded_coeffs[n])
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
    gain = 2**-b_shift
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
    coeffs = apply_biquad_bshift(coeffs, b_shift)
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


def make_biquad_gain(fs, gain_db):
    coeffs = make_biquad_bypass(fs)
    coeffs = apply_biquad_gain(coeffs, gain_db)

    return coeffs


def make_biquad_lowpass(fs, filter_freq, q_factor):

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (filter_freq <= fs/2), 'filter_freq must be less than fs/2'
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

    assert (max(f0, fp) <= fs/2), 'f0 and fp must be less than fs/2'
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


# TODO gain biquad


if __name__ == "__main__":

    fs = 48000

    biquad_1 = biquad(make_biquad_notch(fs, 20, 1), 0, Q_sig=30)
    biquad_2 = biquad(make_biquad_notch(fs, 20, 1), 3, Q_sig=30)
    biquad_3 = biquad(make_biquad_notch(fs, 20, 1), 0, Q_sig=27)
    biquad_4 = biquad(make_biquad_notch(fs, 20, 1), 3, Q_sig=27)
    biquad_5 = biquad(make_biquad_notch(fs, 20, 1), 3, Q_sig=27)

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
    plt.legend(["Q30", "Q30, b_shift 3", "Q27", "Q27, b_shift 3", "double"])
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
