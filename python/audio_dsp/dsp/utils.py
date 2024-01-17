import numpy as np
import scipy.signal as spsig
import math

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


def db_pow2gain(input):
    out = 10**(input/10)
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


def int32(val: int):
    if -2 ** 31 <= val < (2 ** 31 - 1):
        return int(val)
    raise OverflowError


def int64(val: int):
    if -2 ** 63 <= val < (2 ** 63 - 1):
        return int(val)
    raise OverflowError


def int40(val: int):
    # special type for VPU
    if -2 ** 39 <= val < (2 ** 39 - 1):
        return int(val)
    raise OverflowError


def vpu_mult(x1: int, x2: int):

    y = int64(x1*x2)
    y = y + 2**29
    y = int32(y >> 30)

    return y


def vlmaccr(vect1, vect2, out=0):
    for val1, val2 in zip(vect1, vect2):
        out += vpu_mult(val1, val2)

    return int40(out)


def float_to_int32(x):
    return int(round(x*(2**31 - 1)))


def int32_to_float(x):
    return x*2**-31


class float_s32():
    def __init__(self, value):
        if isinstance(value, float):
            self.mant, self.exp = math.frexp(value)
            self.mant = float_to_int32(self.mant)
        elif isinstance(value, int):
            self.mant, self.exp = math.frexp(float(value))
            self.mant = float_to_int32(self.mant)
        elif isinstance(value, list):
            self.mant = value[0]
            self.exp = value[1]
        else:
            TypeError("s32 can only be initialised by float or list of ints [mant, exp]")

        # overflow checks
        self.mant = int32(self.mant)
        self.exp = int32(self.exp)

    def __mul__(self, other_s32):
        if isinstance(other_s32, float_s32):
            return float_s32(float(self) * float(other_s32))
        else:
            raise TypeError("s32 can only be multiplied by s32")

    def __truediv__(self, other_s32):
        if isinstance(other_s32, float_s32):
            return float_s32(float(self) / float(other_s32))
        else:
            raise TypeError("s32 can only be divided by s32")

    def __add__(self, other_s32):
        if isinstance(other_s32, float_s32):
            return float_s32(float(self) + float(other_s32))
        else:
            raise TypeError("s32 can only be added to s32")

    def __sub__(self, other_s32):
        if isinstance(other_s32, float_s32):
            return float_s32(float(self) - float(other_s32))
        else:
            raise TypeError("s32 can only be subtracted from s32")

    def __gt__(self, other_s32):
        if isinstance(other_s32, float_s32):
            return float(self) > float(other_s32)
        else:
            raise TypeError("s32 can only be compared against s32")

    def __lt__(self, other_s32):
        if isinstance(other_s32, float_s32):
            return (float(self) < float(other_s32))
        else:
            raise TypeError("s32 can only be compared against s32")

    def __abs__(self):
        return float_s32([abs(self.mant), self.exp])

    def __float__(self):
        return math.ldexp(int32_to_float(self.mant), self.exp)

    __rmul__ = __mul__


def float_s32_min(x: float_s32, y: float_s32):
    if x > y:
        return y
    else:
        return x


def float_s32_max(x: float_s32, y: float_s32):
    if x > y:
        return x
    else:
        return y


def min_float_s32():
    return float_s32([1, 0])
