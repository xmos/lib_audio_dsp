# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import matplotlib.pyplot as plt

from . import biquad as bq
from . import utils as utils
from audio_dsp.dsp import generic as dspg


class cascaded_biquads_8(dspg.dsp_block):
    def __init__(self, coeffs_list, fs, n_chans, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)
        self.biquads = [None]*8
        for n in range(8):
            if n < len(coeffs_list):
                self.biquads[n] = bq.biquad(coeffs_list[n], fs, n_chans)
            else:
                self.biquads[n] = bq.biquad_bypass(fs, n_chans)

    def process(self, sample):
        y = sample
        for biquad in self.biquads:
            y = biquad.process(y)

        return y

    def process_frame(self, frame):
        y = frame
        for biquad in self.biquads:
            y = biquad.process_frame(y)

        return y

    def process_int(self, sample):
        y = sample
        for biquad in self.biquads:
            y = biquad.process_int(y)

        return y

    def process_frame_int(self, frame):
        y = frame
        for biquad in self.biquads:
            y = biquad.process_frame_int(y)

        return y

    def process_vpu(self, sample):
        y = sample
        for biquad in self.biquads:
            y = biquad.process_vpu(y)

        return y

    def process_frame_vpu(self, frame):
        y = frame
        for biquad in self.biquads:
            y = biquad.process_frame_vpu(y)

        return y

    def freq_response(self, nfft=512):
        f, h_all = self.biquads[0].freq_response(nfft)
        for biquad in self.biquads[1:]:
            _, h = biquad.freq_response(nfft)
            h_all *= h

        return f, h_all

    def reset_state(self):
        for biquad in self.biquads:
            biquad.reset_state()

        return


class butterworth_lowpass(cascaded_biquads_8):
    def __init__(self, fs, n_chans, N, fc):
        coeffs_list = make_butterworth_lowpass(N, fc, fs)
        super().__init__(coeffs_list, fs, n_chans)


class butterworth_highpass(cascaded_biquads_8):
    def __init__(self, fs, n_chans, N, fc):
        coeffs_list = make_butterworth_highpass(N, fc, fs)
        super().__init__(coeffs_list, fs, n_chans)


class parametric_eq_8band(cascaded_biquads_8):
    def __init__(self, fs, n_chans, filter_spec):
        coeffs_list = []
        for spec in filter_spec:
            class_name = f"make_biquad_{spec[0]}"
            class_handle = getattr(bq, class_name)
            coeffs_list.append(class_handle(fs, *spec[1:]))

        super().__init__(coeffs_list, fs, n_chans)


def make_butterworth_lowpass(N, fc, fs):
    # N = filter order (must be even)
    # fc = -3 dB frequency in Hz
    # fs = sample frequency in Hz
    # The function will return N/2 sets of biquad coefficients
    #
    # translated from Neil Robertson's https://www.dsprelated.com/showarticle/1137.php
    #
    # see also https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html

    assert (fc <= fs/2), 'fc must be less than fs/2'
    assert (N % 2 == 0), 'N must be even'

    # Find analog filter poles above the real axis for the low-pass
    ks = np.arange(1, N//2 + 1)
    theta = (2*ks - 1)*np.pi/(2*N)
    pa = -np.sin(theta) + 1j*np.cos(theta)
    # reverse sequence of poles – put high Q last to minimise change of clipping
    pa = np.flip(pa)

    # scale poles in frequency
    Fc = fs/np.pi * np.tan(np.pi*fc/fs)
    pa = pa*2*np.pi*Fc

    # poles in the z plane by bilinear transform
    p = (1 + pa/(2*fs))/(1 - pa/(2*fs))

    coeffs_list = []
    for k in ks:
        # denominator coefficients
        a0 = 1
        a1 = -2*np.real(p[k-1])
        a2 = abs(p[k-1])**2

        # numerator coefficients
        K = (a0 + a1 + a2) / 4
        b0 = K
        b1 = 2*K
        b2 = K

        coeffs = (b0, b1, b2, a0, a1, a2)
        coeffs = bq.normalise_biquad(coeffs)
        coeffs_list.append(coeffs)

    return coeffs_list


def make_butterworth_highpass(N, fc, fs):
    # N = filter order (must be even)
    # fc = -3 dB frequency in Hz
    # fs = sample frequency in Hz
    # The function will return N/2 sets of biquad coefficients
    #
    # translated from from Neil Robertson's https://www.dsprelated.com/showarticle/1137.php and
    # https://www.dsprelated.com/showarticle/1135.php
    #
    # see also https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html

    assert (fc <= fs/2), 'fc must be less than fs/2'
    assert (N % 2 == 0), 'N must be even'

    # Find analog filter poles above the real axis for the low-pass
    ks = np.arange(1, N//2 + 1)
    theta = (2*ks - 1)*np.pi/(2*N)
    pa = -np.sin(theta) + 1j*np.cos(theta)
    # reverse sequence of poles – put high Q last to minimise change of clipping
    pa = np.flip(pa)

    # scale poles in frequency
    Fc = fs/np.pi * np.tan(np.pi*fc/fs)
    # transform to hp poles
    pa = 2*np.pi*Fc/pa

    # poles in the z plane by bilinear transform
    p = (1 + pa/(2*fs))/(1 - pa/(2*fs))

    coeffs_list = []
    for k in ks:
        # denominator coefficients
        a0 = 1
        a1 = -2*np.real(p[k-1])
        a2 = abs(p[k-1])**2

        # numerator coefficients
        K = (a0 - a1 + a2) / 4
        b0 = K
        b1 = -2*K
        b2 = K

        coeffs = (b0, b1, b2, a0, a1, a2)
        coeffs = bq.normalise_biquad(coeffs)
        coeffs_list.append(coeffs)

    return coeffs_list


if __name__ == "__main__":

    fs = 48000
    filter_spec = [['lowpass', 8000, 0.707],
                   ['highpass', 200, 1],
                   ['peaking', 1000, 5, 10]]
    peq = parametric_eq_8band(fs, 1, filter_spec)

    w, response = peq.freq_response()

    fig, figs = plt.subplots(2, 1)
    figs[0].semilogx(w/(2*np.pi)*fs, utils.db(response))
    figs[0].grid()

    figs[1].semilogx(w/(2*np.pi)*fs, np.angle(response))
    figs[1].grid()

    plt.show()

    fc = 6.7
    fs = 100

    a = make_butterworth_highpass(6, fc, fs)

    bq0 = bq.biquad(bq.normalise_biquad(a[0]), fs, 1)
    bq1 = bq.biquad(bq.normalise_biquad(a[1]), fs, 1)
    bq2 = bq.biquad(bq.normalise_biquad(a[2]), fs, 1)

    w, response0 = bq0.freq_response()
    w, response1 = bq1.freq_response()
    w, response2 = bq2.freq_response()
    response = response0*response1*response2

    fig, figs = plt.subplots(2, 1)
    figs[0].plot(w/(2*np.pi)*fs, utils.db(response0))
    figs[0].plot(w/(2*np.pi)*fs, utils.db(response1))
    figs[0].plot(w/(2*np.pi)*fs, utils.db(response2))
    figs[0].plot(w/(2*np.pi)*fs, utils.db(response))
    figs[0].grid()

    figs[1].plot(w/(2*np.pi)*fs, np.angle(response))
    figs[1].grid()

    plt.show()

    pass
