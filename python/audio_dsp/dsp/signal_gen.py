import numpy as np
import scipy.signal as spsig

from audio_dsp.dsp import utils as utils


# These functions return quantized signals, by default to 24b. The output is a
# float scaled between -1 and 1, which can be subsequently scaled to integer
# ranges.


def quantize_signal(signal, precision):
    signal = np.round(signal*2**(precision-1))/2**(precision-1)
    return signal


def sin(fs, length, freq, amplitude, precision=24):
    t = np.arange(fs*length)/fs
    signal = amplitude*np.sin(2*np.pi*freq*t)
    signal = quantize_signal(signal, precision)

    return signal


def log_chirp(fs, length, amplitude, start=20, stop=20000, precision=24):
    t = np.arange(fs*length)/fs
    signal = amplitude*spsig.chirp(t, start, length, stop, 'log', phi=-90)
    signal = quantize_signal(signal, precision)

    return signal


def white_noise(fs, length, amplitude, normal=True, precision=24):
    if normal:
        # normally distributed white noise is unbounded, bound to 6 sigma and
        # scale the variance accordingly by scaling the amplitude
        sigma = 6
        signal = 2/sigma*amplitude*np.random.randn(length*fs)
        signal = np.clip(signal, -1, 1)
    else:
        signal = amplitude*(2*np.random.rand(length*fs) - 1)

    signal = quantize_signal(signal, precision)
    return signal


def pink_filter(fs, fmin=20):

    # parallel 1st order filters at 2 octave spacing
    # after http://cooperbaker.com/home/code/pink%20noise/ with some tweaks to
    # make the gain better

    # signal will be white below fmin, this can help reduce DC fluctuations in
    # the final signal
    n_filters = np.floor(np.emath.logn(4, 48000) - np.emath.logn(4, fmin)).astype(int)

    fc = np.zeros(n_filters)
    a = np.zeros(n_filters)
    g = np.zeros(n_filters)

    for n in range(n_filters):
        fc[n] = (fs / (2*np.pi) - 1) / 4**n
        a[n] = 2*np.pi*(fc[n]/fs)
        g[n] = utils.db2gain(-(6*(n_filters - n - 1)))

    output_g = 2/(np.sum(g))

    return a, g, output_g


def pink_noise(fs, length, amplitude, precision=24):

    # parallel 1st order filters at 2 octave spacing
    # after http://cooperbaker.com/home/code/pink%20noise/ and
    # https://www.musicdsp.org/en/latest/Filters/76-pink-noise-filter.html

    a, g, output_g = pink_filter(fs)
    signal = np.zeros(length*fs)
    y = np.zeros(len(a))
    for n in range(len(signal)):
        w = 2*np.random.rand() - 1
        for nn in range(len(a)):
            y[nn] = a[nn]*w + (1 - a[nn])*y[nn]
            signal[n] += y[nn] * g[nn]
        signal[n] *= output_g

    # scale to full scale, doing it automatically doesn't seem to work well as
    # it depends on fmin
    signal *= 1/np.max(np.abs(signal))

    signal = amplitude*signal
    signal = quantize_signal(signal, precision)
    return signal


def mls(fs, len, amplitude, precision=24):

    len_samples = len*fs
    n_bits = int(np.ceil(np.log2(len_samples + 1)))

    signal, _ = spsig.max_len_seq(n_bits, length=len_samples)
    signal = amplitude*(2*signal - 1)
    signal = quantize_signal(signal, precision)
    return signal


if __name__ == "__main__":
    x = white_noise(48000, 1, 1)

    x2 = white_noise(48000, 1, 1, normal=False)

    m = pink_noise(48000, 1, 1)

    pass
