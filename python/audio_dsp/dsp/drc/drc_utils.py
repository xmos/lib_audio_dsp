# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Generic utilities for the dynamic range control DSP blocks."""

from audio_dsp.dsp import utils as utils
import numpy as np
from math import sqrt, isqrt
import warnings

from audio_dsp.dsp.types import float32


FLT_MIN = np.finfo(float).tiny

# Q format for the drc alphas and gains
Q_alpha = 31


def calculate_rms_threshold(threshold_db, Q_sig) -> tuple[float, int]:
    """
    Calculate the linear RMS threshold in floating and fixed point from a
    target threshold in decibels.
    If the threshold is higher than representable in the fixed point
    format, it is saturated.
    """
    return calculate_threshold(threshold_db, Q_sig, power=True)


def calculate_peak_threshold(threshold_db, Q_sig) -> tuple[float, int]:
    """
    Calculate the linear peak threshold in floating and fixed point from a
    target threshold in decibels.
    If the threshold is higher than representable in the fixed point
    format, it is saturated.
    """
    return calculate_threshold(threshold_db, Q_sig, power=False)


def calculate_threshold(threshold_db, Q_sig, power=False) -> tuple[float, int]:
    """
    Calculate the linear threshold in floating and fixed point from a
    target threshold in decibels.
    If the threshold is higher than representable in the fixed point
    format, it is saturated.
    """
    if power:
        threshold = utils.db_pow2gain(threshold_db)
    else:
        threshold = utils.db2gain(threshold_db)

    threshold = utils.saturate_float(threshold, Q_sig)

    if power:
        new_threshold_db = utils.db_pow(threshold)
    else:
        new_threshold_db = utils.db(threshold)

    if threshold_db != new_threshold_db:
        warnings.warn(
            "Threshold %.2f not representable in Q format Q%d, saturating to %.2f"
            % (threshold_db, Q_sig, new_threshold_db),
            UserWarning,
        )

    threshold_int = utils.float_to_int32(threshold, Q_sig)

    # this avoids division by zero for expanders
    threshold_int = max(1, threshold_int)

    return threshold, threshold_int


def alpha_from_time(attack_or_release_time, fs):
    """
    Calculate the exponential moving average time constant from an
    attack/release time in seconds.

    Attack times simplified from McNally, seem pretty close.
    Assumes the time constant of a digital filter is the -3 dB
    point where abs(H(z))**2 = 0.5.

    This is also approximately the time constant of a first order
    system, `alpha = 1 - exp(-T/tau)`, where `T` is the sample period
    and `tau` is the time constant.

    attack/release time can't be faster than the length of 2
    samples, and alpha can't be greater than 1. This function will
    saturate to those values.
    """
    if attack_or_release_time < 0:
        warnings.warn(
            "Attack/release time must not be negative. For the fastest possible "
            "attack/release time, use zero. Time set to zero",
            UserWarning,
        )
        attack_or_release_time = 0

    T = 1 / fs
    alpha = 2 * T / (attack_or_release_time + FLT_MIN)

    if alpha > 1:
        alpha = 1
        warnings.warn(
            "Attack or release time too fast for sample rate, setting as fast as possible.",
            UserWarning,
        )

    alpha_int = utils.int32(round(alpha * 2**31)) if alpha != 1.0 else utils.int32(2**31 - 1)

    # This is possible if alpha > (4/fs)*(2**31), which is 49.7 hours @ 48kHz,
    # in which case you should probably use a lower sample rate.
    if alpha_int <= 0:
        warnings.warn(
            "alpha not > 0, this is possible if attack/release time > (4/fs)*(2**31)."
            "Setting alpha_int to 0 (no smoothing)",
            UserWarning,
        )
        alpha_int = 0

    return alpha, alpha_int


def rms_compressor_slope_from_ratio(ratio):
    """Convert a compressor ratio to the slope, where the slope is
    defined as (1 - 1 / ratio) / 2.0. The division by 2 compensates for
    the RMS envelope detector returning the RMS².
    """
    if ratio < 1:
        warnings.warn("Compressor ratio must be >= 1, setting ratio to 1", UserWarning)
        ratio = 1

    slope = (1 - 1 / ratio) / 2.0
    slope_f32 = float32(slope)
    return slope, slope_f32


def peak_expander_slope_from_ratio(ratio):
    """Convert an expander ratio to the slope, where the slope is
    defined as (1 - ratio).
    """
    if ratio < 1:
        warnings.warn("Expander ratio must be >= 1, setting ratio to 1", UserWarning)
        ratio = 1

    slope = 1 - ratio
    slope_f32 = float32(slope)
    return slope, slope_f32


def calc_ema_xcore(x, y, alpha):
    """Calculate fixed-point exponential moving average, given that alpha is in Q_alpha format."""
    acc = int(x) << Q_alpha
    mul = utils.int32(y - x)
    acc += mul * alpha
    x = utils.int32_mult_sat_extract(acc, 1, Q_alpha)
    return x


def apply_gain_xcore(sample: int, gain: int) -> int:
    """Apply the gain to a sample using fixed-point math. Assumes that gain is in Q_alpha format."""
    acc = 1 << (Q_alpha - 1)
    acc += sample * gain
    y = utils.int32_mult_sat_extract(acc, 1, Q_alpha)
    return y


def limiter_peak_gain_calc(envelope, threshold, slope=None):
    """Calculate the float gain for the current sample."""
    new_gain = threshold / envelope
    new_gain = min(1, new_gain)
    return new_gain


def limiter_peak_gain_calc_xcore(envelope_int, threshold_int, slope=None):
    """Calculate the int gain for the current sample."""
    if threshold_int >= envelope_int:
        new_gain_int = utils.int32(0x7FFFFFFF)
    else:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
    return new_gain_int


def limiter_rms_gain_calc(envelope, threshold, slope=None):
    """Calculate the float gain for the current sample.

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain.

    """
    new_gain = sqrt(threshold / envelope)
    new_gain = min(1, new_gain)
    return new_gain


def limiter_rms_gain_calc_xcore(envelope_int, threshold_int, slope=None):
    """Calculate the int gain for the current sample.

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain.

    """
    if threshold_int >= envelope_int:
        new_gain_int = utils.int32(0x7FFFFFFF)
    else:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
        new_gain_int = utils.int32(isqrt(new_gain_int * 2**31))
    return new_gain_int


def compressor_rms_gain_calc(envelope, threshold, slope=None):
    """Calculate the float gain for the current sample.

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
    # if envelope below threshold, apply unity gain, otherwise scale
    # down
    new_gain = (threshold / envelope) ** slope
    new_gain = min(1, new_gain)
    return new_gain


def compressor_rms_gain_calc_xcore(envelope_int, threshold_int, slope_f32=None):
    """Calculate the int gain for the current sample.

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
    # if envelope below threshold, apply unity gain, otherwise scale
    # down
    int32_max_as_f32 = float32(np.nextafter(2**31, 0, dtype=np.float32))

    if slope_f32 > float32(0) and threshold_int < envelope_int:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
        new_gain_int = (
            (float32(new_gain_int * 2**-31) ** slope_f32) * int32_max_as_f32
        ).as_int32()
    else:
        new_gain_int = utils.int32(0x7FFFFFFF)

    return new_gain_int


def noise_gate_gain_calc(envelope, threshold, slope=None):
    """Calculate the float gain for the current sample."""
    if envelope < threshold:
        new_gain = 0
    else:
        new_gain = 1
    return new_gain


def noise_gate_gain_calc_xcore(envelope_int, threshold_int, slope_int=None):
    """Calculate the int gain for the current sample."""
    if envelope_int < threshold_int:
        new_gain_int = utils.int32(0)
    else:
        new_gain_int = utils.int32(2**31 - 1)
    return new_gain_int


def noise_suppressor_expander_gain_calc(envelope, threshold, slope):
    """Calculate the float gain for the current sample."""
    # if envelope above threshold, apply unity gain, otherwise scale
    # down
    new_gain = (threshold / envelope) ** slope
    new_gain = min(1, new_gain)
    return new_gain


def noise_suppressor_expander_gain_calc_xcore(envelope_int, threshold_int, slope_f32):
    """Calculate the int gain for the current sample."""
    # if envelope above threshold, apply unity gain, otherwise scale
    # down
    # note this is rearranged to (envelope / threshold) ** -slope in order
    # to be similar to the compressor implementation, which also allows
    # 1/threshold to be precomputed
    invt = utils.int64(((1 << 63) - 1) // threshold_int)
    if -slope_f32 > float32(0) and threshold_int > envelope_int:
        # this looks a bit scary, but as long as envelope < threshold,
        # it can't overflow
        new_gain_int = utils.int64(envelope_int * invt)
        new_gain_int = ((float32(new_gain_int * 2**-63) ** -slope_f32) * float32(2**31)).as_int32()
    else:
        new_gain_int = utils.int32(0x7FFFFFFF)

    return new_gain_int
