# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Signal generator DSP utilities."""

import numpy as np
import numpy.lib.scimath as emath
import scipy.signal as spsig

from audio_dsp.dsp import utils as utils


# These functions return quantized signals, by default to 24b. The
# output is a float scaled between -1 and 1, which can be subsequently
# scaled to integer ranges.


def quantize_signal(signal: np.ndarray, precision: int) -> np.ndarray:
    """Quantizes the input signal to the specified precision.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to be quantized.
    precision : int
        The number of bits used for quantization precision.

    Returns
    -------
    np.ndarray
        The quantized signal.

    """
    signal = np.round(signal * (2 ** (precision - 1) - 1)) / 2 ** (precision - 1)
    return signal


def sin(fs: int, length: float, freq: float, amplitude: float, precision: int = 24) -> np.ndarray:
    """
    Generate a quantized sinusoidal signal.

    Parameters
    ----------
    fs : int
        The sampling frequency in Hz.
    length : float
        The duration of the signal in seconds.
    freq : float
        The frequency of the sinusoid in Hz.
    amplitude : float
        The amplitude of the sinusoid.
    precision : int, optional
        The precision of the quantization in bits, by default 24.

    Returns
    -------
    np.ndarray
        The generated sinusoidal signal.
    """
    t = np.arange(int(fs * length)) / fs
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signal = quantize_signal(signal, precision)

    return signal


def cos(fs: int, length: float, freq: float, amplitude: float, precision: int = 24) -> np.ndarray:
    """
    Generate a quantized cosine signal.

    Parameters
    ----------
    fs : int
        The sampling frequency in Hz.
    length : float
        The duration of the signal in seconds.
    freq : float
        The frequency of the cosine signal in Hz.
    amplitude : float
        The amplitude of the cosine signal.
    precision : int, optional
        The precision of the quantized signal, in number of bits.
        Default is 24.

    Returns
    -------
    np.ndarray
        The generated cosine signal as a numpy array.
    """
    t = np.arange(int(fs * length)) / fs
    signal = amplitude * np.cos(2 * np.pi * freq * t)
    signal = quantize_signal(signal, precision)

    return signal


def square(
    fs: int, length: float, freq: float, amplitude: float, precision: int = 24
) -> np.ndarray:
    """
    Generate a quantized square wave signal.

    Parameters
    ----------
    fs : int
        The sampling frequency in Hz.
    length : float
        The duration of the signal in seconds.
    freq : float
        The frequency of the cosine signal in Hz.
    amplitude : float
        The amplitude of the cosine signal.
    precision : int, optional
        The precision of the quantized signal, in number of bits.
        Default is 24.

    Returns
    -------
    np.ndarray
        The generated cosine signal as a numpy array.
    """
    t = np.arange(int(fs * length)) / fs
    signal = amplitude * spsig.square(2 * np.pi * freq * t)
    signal = quantize_signal(signal, precision)

    return signal


def log_chirp(
    fs: int,
    length: float,
    amplitude: float,
    start: float = 20,
    stop: float = 20000,
    precision: int = 24,
) -> np.ndarray:
    """
    Generate a quantized logarithmic chirp signal.

    Parameters
    ----------
    fs : int
        The sample rate of the signal.
    length : float
        The duration of the signal in seconds.
    amplitude : float
        The amplitude of the signal.
    start : float, optional
        The starting frequency of the chirp signal in Hz. Default is
        20 Hz.
    stop : float, optional
        The ending frequency of the chirp signal in Hz. Default is
        20000 Hz.
    precision : int, optional
        The precision of the quantization in bits. Default is 24 bits.

    Returns
    -------
    np.ndarray
        The generated logarithmic chirp signal as a NumPy array.
    """
    t = np.arange(int(fs * length)) / fs
    signal = amplitude * spsig.chirp(t, start, length, stop, "log", phi=-90)
    signal = quantize_signal(signal, precision)

    return signal


def white_noise(
    fs: int, length: float, amplitude: float, normal: bool = True, precision: int = 24
) -> np.ndarray:
    """
    Generate a quantized white noise signal.

    Parameters
    ----------
    fs : int
        The sampling frequency of the signal.
    length : float
        The duration of the signal in seconds.
    amplitude : float
        The amplitude of the signal.
    normal : bool, optional
        If True, generate normally distributed white noise. If False,
        generate uniformly distributed white noise.
        Default is True.
    precision : int, optional
        The precision (number of bits) used for quantizing the signal.
        Default is 24.

    Returns
    -------
    np.ndarray
        The generated white noise signal as a NumPy array.
    """
    if normal:
        # normally distributed white noise is unbounded, bound to 6 sigma and
        # scale the variance accordingly by scaling the amplitude
        sigma = 6
        signal = 2 / sigma * amplitude * np.random.randn(round(length * fs))
        signal = np.clip(signal, -1, 1)
    else:
        signal = amplitude * (2 * np.random.rand(round(length * fs)) - 1)

    signal = quantize_signal(signal, precision)
    return signal


def pink_filter(fs: int, fmin: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a pink noise filterbank.

    This function generates a pink noise filterbank by implementing
    parallel 1st order filters at 2 octave spacing. The signal will be
    white below fmin, which can help reduce DC fluctuations in the final
    signal.

    After http://cooperbaker.com/home/code/pink%20noise/, with some
    tweaks to make the gain better.

    Parameters
    ----------
    fs : int
        The sampling frequency.
    fmin : int, optional
        The minimum frequency below which the signal will be white, by
        default 20.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        A tuple containing the filter coefficients (a), gains (g), and
        output gain (output_g).
    """
    n_filters = np.floor(emath.logn(4, 48000) - emath.logn(4, fmin)).astype(int)

    fc = np.zeros(n_filters)
    a = np.zeros(n_filters)
    g = np.zeros(n_filters)

    for n in range(n_filters):
        fc[n] = (fs / (2 * np.pi) - 1) / 4**n
        a[n] = 2 * np.pi * (fc[n] / fs)
        g[n] = utils.db2gain(-(6 * (n_filters - n - 1)))

    output_g = float(2 / (np.sum(g)))

    return a, g, output_g


def pink_noise(fs: int, length: float, amplitude: float, precision: int = 24) -> np.ndarray:
    """
    Generate a quantized pink noise signal.

    This function generates pink noise signal using parallel 1st order
    filters at 2 octave spacing.

    Parameters
    ----------
    fs : int
        The sample rate of the generated signal.
    length : float
        The length of the generated signal in seconds.
    amplitude : float
        The amplitude of the generated signal.
    precision : int, optional
        The precision of the quantized signal, by default 24.

    Returns
    -------
    np.ndarray
        The generated pink noise signal.

    References
    ----------
    - http://cooperbaker.com/home/code/pink%20noise/
    - https://www.musicdsp.org/en/latest/Filters/76-pink-noise-filter.html
    """
    a, g, output_g = pink_filter(fs)
    signal = np.zeros(int(length * fs))
    y = np.zeros(len(a))
    for n in range(len(signal)):
        w = 2 * np.random.rand() - 1
        for nn in range(len(a)):
            y[nn] = a[nn] * w + (1 - a[nn]) * y[nn]
            signal[n] += y[nn] * g[nn]
        signal[n] *= output_g

    # scale to full scale, doing it automatically doesn't seem to work well as
    # it depends on fmin
    signal *= 1 / np.max(np.abs(signal))

    signal = amplitude * signal
    signal = quantize_signal(signal, precision)
    return signal


def mls(fs: int, length: float, amplitude: float, precision: int = 24) -> np.ndarray:
    """
    Generate a quantized Maximum Length Sequence (MLS) signal.

    Parameters
    ----------
    fs : int
        The sampling frequency of the signal.
    length : float
        The duration of the signal in seconds.
    amplitude : float
        The amplitude of the signal.
    precision : int, optional
        The number of bits used for quantization, by default 24.

    Returns
    -------
    numpy.ndarray
        The generated MLS signal.

    """
    len_samples = length * fs
    n_bits = int(np.ceil(np.log2(len_samples + 1)))

    signal, _ = spsig.max_len_seq(n_bits, length=len_samples)
    signal = amplitude * (2 * signal - 1)
    signal = quantize_signal(signal, precision)
    return signal


if __name__ == "__main__":
    x = white_noise(48000, 1, 1)

    x2 = white_noise(48000, 1, 1, normal=False)

    m = pink_noise(48000, 1, 1)

    pass
