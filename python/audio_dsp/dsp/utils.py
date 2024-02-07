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


def uq_2_30(val: int):
    # special type for unsigned Q2.30 format, used by EWM
    if 0 < val < (2 ** 32):
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
    return int32(round(x*(2**31 - 1)))


def int32_to_float(x):
    return x*2**-31


class float_s32():
    def __init__(self, value, Q_sig=None):
        self.Q_sig = Q_sig
        if Q_sig and isinstance(value, float):
            self.mant = int32(round(value * 2**Q_sig))
            self.exp = -Q_sig
        elif isinstance(value, float) or isinstance(value, np.float32):
            self.mant, self.exp = math.frexp(value)
            self.mant = float_to_int32(self.mant)
            self.exp -= 31
        elif isinstance(value, int):
            self.mant, self.exp = math.frexp(float(value))
            self.mant = float_to_int32(self.mant)
            self.exp -= 31
        elif isinstance(value, list):
            self.mant = int32(value[0])
            self.exp = int32(value[1])
        else:
            TypeError("s32 can only be initialised by float or list of ints [mant, exp]")

        # overflow checks
        self.mant = int32(self.mant)
        self.exp = int32(self.exp)

    def __mul__(self, other_s32):
        if isinstance(other_s32, float_s32):

            # vect_s32_mul_prepare
            b_hr = hr_s32(self)
            c_hr = hr_s32(other_s32)
            total_hr = b_hr + c_hr

            if total_hr == 0:
                b_shr = 1
                c_shr = 1
            elif total_hr == 1:
                b_shr = 1 if b_hr == 0 else 0
                c_shr = 1 if c_hr == 0 else 0
            elif b_hr == 0:
                b_shr = 0
                c_shr = 2 - total_hr
            elif c_hr == 0:
                b_shr = 2 - total_hr
                c_shr = 0
            else:
                b_shr = 1 - b_hr
                c_shr = 1 - c_hr

            res_exp = self.exp + other_s32.exp + b_shr + c_shr + 30

            # vect_s32_mul
            B = ashr32(self.mant, b_shr)
            C = ashr32(other_s32.mant, c_shr)

            A = vpu_mult(B, C)

            return float_s32([A, res_exp])
        else:
            raise TypeError("s32 can only be multiplied by s32")

    def __truediv__(self, other_s32):
        if isinstance(other_s32, float_s32):
            # float_s32_div and s32_inverse
            b_hr = hr_s32(other_s32)
            scale = 2*30 - b_hr
            dividend = 1 << scale
            t = float_s32([dividend / other_s32.mant, -scale - other_s32.exp])

            return self.__mul__(t)

        else:
            raise TypeError("s32 can only be divided by s32")

    def __add__(self, other_s32):
        if isinstance(other_s32, float_s32):
            # from float_s32_add in lib_xcore_math
            x_hr = hr_s32(self)
            y_hr = hr_s32(other_s32)
            x_min_exp = self.exp - x_hr
            y_min_exp = other_s32.exp - y_hr

            res_exp = max(x_min_exp, y_min_exp) + 1

            x_shr = res_exp - self.exp
            y_shr = res_exp - other_s32.exp

            mant = ashr32(self.mant, x_shr) + ashr32(other_s32.mant, y_shr)

            return float_s32([mant, res_exp])
        else:
            raise TypeError("s32 can only be added to s32")

    def __sub__(self, other_s32):
        if isinstance(other_s32, float_s32):
            # from flaot_s32_sub
            x_hr = hr_s32(self)
            y_hr = hr_s32(other_s32)
            x_min_exp = self.exp - x_hr
            y_min_exp = other_s32.exp - y_hr

            res_exp = max(x_min_exp, y_min_exp) + 1

            x_shr = res_exp - self.exp
            y_shr = res_exp - other_s32.exp

            mant = ashr32(self.mant, x_shr) - ashr32(other_s32.mant, y_shr)

            return float_s32([mant, res_exp])
        else:
            raise TypeError("s32 can only be subtracted from s32")

    def __gt__(self, other_s32):
        if isinstance(other_s32, float_s32):
            delta = self.__sub__(other_s32)
            return delta.mant > 0
        else:
            raise TypeError("s32 can only be compared against s32")

    def __lt__(self, other_s32):
        if isinstance(other_s32, float_s32):
            delta = self.__sub__(other_s32)
            return delta.mant < 0
        else:
            raise TypeError("s32 can only be compared against s32")

    def __abs__(self):
        return float_s32([abs(self.mant), self.exp])

    def __float__(self):
        # add 31 here python expects a float mantissa < 1, but we use int32
        return math.ldexp(int32_to_float(self.mant), self.exp + 31)

    __rmul__ = __mul__


def hr_s32(x: float_s32):
    # calculate number of leading zeros on the mantissa
    return 31 - x.mant.bit_length()


def ashr32(x, shr):
    # right shift, with negative values shifting left
    if shr >= 0:
        return x >> shr
    else:
        return x << -shr


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


def float_s32_ema(x: float_s32, y: float_s32, alpha: int):
    # this is an implementation of float_s32_ema in lib_xcore_math
    t = float_s32([alpha, -30])
    s = float_s32([2**30 - alpha, -30])

    output = (x*t) + (y*s)

    return output

def float_s32_to_fixed(val : float_s32, out_exp : int):
    shr = out_exp - val.exp
    return ashr32(val.mant, shr)


def float_s32_use_exp(val : float_s32, out_exp : int):
    val.mant = float_s32_to_fixed(val, out_exp)
    val.exp = out_exp
    return val


def frame_signal(signal, buffer_len, step_size):
    n_samples = signal.shape[1]
    n_frames = int(np.floor((n_samples - buffer_len)/step_size) + 1)
    output = []

    for n in range(n_frames):
        output.append(np.copy(signal[:, n*step_size:n*step_size + buffer_len]))

    return output
