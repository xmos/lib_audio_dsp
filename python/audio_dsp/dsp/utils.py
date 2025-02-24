# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Utility functions used by DSP blocks."""

import numpy as np
import scipy.signal as spsig
import math
import warnings
import typing

from audio_dsp.dsp.types import float32, float_s32

FLT_MIN = np.finfo(float).tiny


class OverflowWarning(Warning):
    """A warning for integer overflows."""

    pass


class SaturationWarning(Warning):
    """A warning for when a value has been saturated to prevent overflow."""

    pass


def db(input):
    """Convert an amplitude to decibels (20*log10(abs(x)))."""
    out = 20 * np.log10(np.abs(input) + FLT_MIN)
    return out


def db_pow(input):
    """Convert a power to decibels (10*log10(abs(x)))."""
    out = 10 * np.log10(np.abs(input) + FLT_MIN)
    return out


def db2gain(input):
    """Convert from decibels to amplitude (10^(x/20))."""
    out = 10 ** (input / 20)
    return out


def db2gain_f32(input):
    """Convert from decibels to amplitude in float32 (10^(x/20))."""
    out = float32(10) ** (float32(input) / float32(20))
    return out


def db_pow2gain(input):
    """Convert from decibels to power (10^(x/10))."""
    out = 10 ** (input / 10)
    return out


def leq_smooth(x, fs, T):
    """Calculate the Leq (equivalent continuous sound level) of a signal."""
    len_x = x.shape[0]
    win_len = int(fs * T)
    win_count = len_x // win_len
    len_y = win_len * win_count

    y = np.reshape(x[:len_y], (win_len, win_count), "F")

    leq = 10 * np.log10(np.mean(y**2.0, axis=0) + FLT_MIN)
    t = np.arange(win_count) * T

    return t, leq


def envelope(x, N=None):
    """Calculate the envelope of a signal using the Hilbert transform."""
    y = spsig.hilbert(x, N)
    return np.abs(y)  # pyright: ignore, not sure why pyright hates this


def int32(val: float | int) -> int:
    """32 bit integer type.
    Integers in Python are larger than 64b, so checks the value is
    within the valid range.
    This function overflows if val is outside the range of int32.
    """
    if -(1 << 31) <= val <= ((1 << 31) - 1):
        return int(val)
    else:
        warnings.warn("Overflow occurred", OverflowWarning)
        return int(((val + (1 << 31)) % (1 << 32)) - (1 << 31))


def saturate_float(val: float, Q_sig: int) -> float:
    """Saturate a single floating point number to the max/min values of
    a given Q format.
    """
    max_flt = float((Q_max(31)) / Q_max(Q_sig))
    min_flt = -float((Q_max(31) + 1) / Q_max(Q_sig))
    if min_flt <= val <= max_flt:
        return val
    elif val < min_flt:
        warnings.warn("Saturation occurred", SaturationWarning)
        return min_flt
    else:
        warnings.warn("Saturation occurred", SaturationWarning)
        return max_flt


def saturate_float_array(val: np.ndarray, Q_sig: int) -> np.ndarray:
    """Saturate a floating point array to the max/min values of
    a given Q format.
    """
    max_flt = float((Q_max(31)) / Q_max(Q_sig))
    min_flt = -float((Q_max(31) + 1) / Q_max(Q_sig))

    if np.any(val < min_flt) or np.any(val > max_flt):
        warnings.warn("Saturation occurred", SaturationWarning)

    val[val > max_flt] = max_flt
    val[val < min_flt] = min_flt

    return val


def saturate_int32(val: int) -> int:
    """Saturate int32 to int32max or min."""
    if -(1 << 31) <= val <= ((1 << 31) - 1):
        return int(val)
    elif val < -(1 << 31):
        warnings.warn("Saturation occurred", SaturationWarning)
        return int(-(1 << 31))
    else:
        warnings.warn("Saturation occurred", SaturationWarning)
        return int(((1 << 31) - 1))


def saturate_int32_vpu(val: int) -> int:
    """Symmetrically saturate int32 to ±int32max. This emulates XS3 VPU
    saturation.
    """
    if -Q_max(31) <= val <= Q_max(31):
        return int(val)
    elif val < -Q_max(31):
        warnings.warn("Saturation occurred", SaturationWarning)
        return -Q_max(31)
    else:
        warnings.warn("Saturation occurred", SaturationWarning)
        return Q_max(31)


def int34(val: float):
    """34 bit integer type. This is used in the VPU multiplication
    product after shifting, before accumulating into an int40.
    Integers in Python are larger than 64b, so checks the value is
    within the valid range.
    """
    if (-(1 << 33)) <= val <= ((1 << 33) - 1):
        return int(val)
    raise OverflowError


def int64(val: float):
    """64 bit integer type.
    Integers in Python are larger than 64b, so checks the value is
    within the valid range.
    """
    if (-(1 << 63)) <= val <= ((1 << 63) - 1):
        return int(val)
    raise OverflowError


def int40(val: int):
    """40 bit integer type. This emulates the XS3 VPU accumulators.
    Integers in Python are larger than 64b, so checks the value is
    within the valid range.
    """
    if (-(1 << 39)) <= val <= ((1 << 39) - 1):
        return int(val)
    raise OverflowError


def uq_2_30(val: int):
    """Unsigned Q2.30 integer format, used by EWM (exponentially
    weighted moving average).

    Integers in Python are larger than int64, so checks the value is
    within the valid range.
    """
    if 0 <= val < (1 << 32):
        return int(val)
    raise OverflowError


def vpu_mult(x1: int, x2: int):
    """Multiply 2 int32 values and apply a 30bit shift with rounding.
    This emulates the XS3 VPU multiplicaions.
    """
    y = int64(x1 * x2)
    y = y + (1 << 29)
    y = int34(y >> 30)

    return y


def int32_mult_sat_extract(x1: int, x2: int, Q: int):
    """Multiply two int32s, shifting the result right by Q.
    If the shifted result will exceed ((1<<31) - 1), saturate before
    shifting.
    """
    y = int64(x1 * x2)
    if y > Q_max(31 + Q):
        warnings.warn("Saturation occurred", SaturationWarning)
        y = Q_max(31 + Q)
    elif y < -Q_max(31 + Q) - 1:
        warnings.warn("Saturation occurred", SaturationWarning)
        y = -Q_max(31 + Q) - 1
    y = int32(y >> Q)

    return y


def saturate_int64_to_int32(x: int):
    """Convert an int64 value to int32, saturating if it is greater than
    ((1<<31) - 1).
    """
    if x > ((1 << 31) - 1):
        warnings.warn("Saturation occurred", SaturationWarning)
        return (1 << 31) - 1
    elif x < -(1 << 31):
        warnings.warn("Saturation occurred", SaturationWarning)
        return -(1 << 31)
    else:
        return x


def vlmaccr(vect1, vect2, out=0):
    """Multiply-accumulate 2 int32 vectors into an int40 result.
    This emulates the XS3 VPU behaviour.
    """
    for val1, val2 in zip(vect1, vect2):
        out += vpu_mult(val1, val2)

    return int40(out)


def Q_max(Q_format: int) -> int:
    """Return the maximum value for a give n Q format, i.e.
    ``(1 << Q_format) - 1``.
    """
    return int((1 << Q_format) - 1)


def float_to_fixed(x: float, Q_sig=31) -> int:
    """Round and scale a floating point number to an int32 in a given
    Q format.
    """
    return int32(round(x * Q_max(Q_sig)))


def fixed_to_float(x: int, Q_sig: int = 31) -> float:
    """Convert an int32 number to floating point, given its Q format."""
    return float(x) / float(Q_max(Q_sig))


def float_to_fixed_array(x: np.ndarray, Q_sig=31) -> np.ndarray:
    """Round and scale a floating point array to an int32 in a given
    Q format.
    """
    return (np.round(x * Q_max(Q_sig))).astype(int)


def fixed_to_float_array(x: np.ndarray, Q_sig: int = 31) -> np.ndarray:
    """Convert an int32 array to floating point, given its Q format."""
    return x.astype(float) / float(Q_max(Q_sig))


def float_to_int32(x, Q_sig=31) -> int:
    """Round and scale a floating point number to an int32 in a given
    Q format.
    """
    return int32(round(x * (1 << Q_sig)))


def float_list_to_int32(x, Q_sig=31) -> typing.List[int]:
    """Round and scale a list of floating point numbers to a list of
    int32s in a given Q format.
    """
    return [int32(round(item * (1 << Q_sig))) for item in x]


def int32_to_float(x: int, Q_sig: int = 31) -> float:
    """Convert an int32 number to floating point, given its Q format."""
    # Note this means the max value is 0.99999999953
    return float(x) / float(1 << Q_sig)


def hr_s32(x: float_s32):
    """Calculate number of leading zeros on the mantissa of a float_s32."""
    assert isinstance(x.mant, int)
    return 31 - x.mant.bit_length()


def ashr32(x, shr):
    """Right shift, with negative values shifting left."""
    if shr >= 0:
        return x >> shr
    else:
        return x << -shr


def float_s32_ema(x: float_s32, y: float_s32, alpha: int):
    """Calculate the exponential moving average of a float_s32.
    This is an implementation of float_s32_ema in lib_xcore_math.
    """
    t = float_s32([alpha, -30])
    s = float_s32([(1 << 30) - alpha, -30])

    output = (x * t) + (y * s)

    return output


def float_s32_to_fixed(val: float_s32, out_exp: int):
    """Convert a float_32 value to fixed point. Shift the mantissa of
    val by the difference between the current expoent and the desired
    exponent to get the correct fixed point scaling.
    """
    shr = out_exp - val.exp
    return ashr32(val.mant, shr)


def float_s32_use_exp(val: float_s32, out_exp: int):
    """Set the exponent of a float_s32 to a specific value."""
    val.mant = float_s32_to_fixed(val, out_exp)
    val.exp = out_exp
    return val


def frame_signal(signal, buffer_len, step_size):
    """Split a signal into overlapping frames. Each frame is buffer_len
    long, and there are step_size samples between frames.
    """
    n_samples = signal.shape[1]
    n_frames = int(np.floor((n_samples - buffer_len) / step_size) + 1)
    output = []

    for n in range(n_frames):
        output.append(np.copy(signal[:, n * step_size : n * step_size + buffer_len]))

    return output


def check_time_units(units: str) -> str:
    """Check a time units in seconds/milliseconds/samples match the expected values.

    Parameters
    ----------
    units : {"samples", "ms", "s"}
        Desired units of time. If invalid units are given, samples are assumed

    Returns
    -------
    Used ``units`` of time
    """
    units = units.lower()
    if units not in ["samples", "ms", "s"]:
        warnings.warn("Time units not recognised, assuming samples")
        units = "samples"

    return units


def time_to_samples(fs, time: float, units: str) -> int:
    """Convert a time in seconds/milliseconds/samples to samples for a
    given sampling frequency.

    Parameters
    ----------
    fs : float
        sampling frequency
    time : float
        desired time in units
    units : {"samples", "ms", "s"}
        desired units of time

    Returns
    -------
    Time converted from ``units`` to samples at fs.

    Raises
    ------
    ValueError
        If time is negative or invalid time units are passed.

    """
    if time < 0:
        warnings.warn("Delay must be positive, setting delay to 0", UserWarning)
        time = 0

    units = check_time_units(units)

    if units == "ms":
        time = int(time * fs / 1000)
    elif units == "s":
        time = int(time * fs)
    elif units == "samples":
        time = int(time)
    else:
        raise ValueError("Units must be 'samples', 'ms' or 's'")
    return time


def quantize_array(coefs: np.ndarray, exp: float):
    """
    Quantise an np.ndarray with exponent exp.

    Parameters
    ----------
    coefs : np.ndarray
        Array of floats to be quanntised
    exp : float
        Exponent to use for the quantisation

    Returns
    -------
    np.array
         Array of ints
    """
    quantised = np.rint(np.ldexp(coefs, exp))
    quantised_and_clipped = np.clip(quantised, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    assert np.allclose(quantised, quantised_and_clipped)
    return np.array(quantised_and_clipped, dtype=np.int64)
