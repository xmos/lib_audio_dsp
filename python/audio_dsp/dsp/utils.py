import numpy as np
import scipy.signal as spsig

FLT_MIN = np.finfo(float).tiny


def db(input):
    out = 20*np.log10(np.abs(input) + FLT_MIN)
    return out


def db_pow(input):
    out = 10*np.log10(np.abs(input) + FLT_MIN)
    return out


def db2gain(input):
    out = 10**(input/20)
    return out


def leq_smooth(x, fs, T):
    len_x = x.shape[0]
    win_len = int(fs * T)
    win_count = len_x // win_len
    len_y = win_len * win_count

    y = np.reshape(x[:len_y], (win_len, win_count), 'F')

    leq = 10 * np.log10(np.mean(y ** 2.0, axis=0) + FLT_MIN)
    t = np.arange(win_count) * T

    return t, leq


def envelope(x, N=None):
    y = spsig.hilbert(x, N)
    return np.abs(y)
