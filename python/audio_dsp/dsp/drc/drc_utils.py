# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from audio_dsp.dsp import utils as utils
import numpy as np
from math import sqrt, isqrt

from audio_dsp.dsp.types import float32


FLT_MIN = np.finfo(float).tiny

# Q format for the drc alphas and gains
Q_alpha = 31


def alpha_from_time(attack_or_release_time, fs):
    # Attack times simplified from McNally, seem pretty close.
    # Assumes the time constant of a digital filter is the -3 dB
    # point where abs(H(z))**2 = 0.5.

    # This is also approximately the time constant of a first order
    # system, `alpha = 1 - exp(-T/tau)`, where `T` is the sample period
    # and `tau` is the time constant.

    # attack/release time can't be faster than the length of 2
    # samples, and alpha can't be greater than 1

    T = 1 / fs
    alpha = 2 * T / (attack_or_release_time + FLT_MIN)

    if alpha > 1:
        alpha = 1
        Warning("Attack or release time too fast for sample rate, setting as fast as possible.")

    # I don't think this is possible, but let's make sure!
    assert alpha > 0

    alpha_int = utils.int32(round(alpha * 2**31)) if alpha != 1.0 else utils.int32(2**31 - 1)
    assert alpha_int > 0
    return alpha, alpha_int


def calc_ema_xcore(x, y, alpha):
    """Calculate fixed-point exponential moving average, given that alpha is in Q_alpha format"""
    acc = int(x) << Q_alpha
    mul = utils.int32(y - x)
    acc += mul * alpha
    x = utils.int32_mult_sat_extract(acc, 1, Q_alpha)
    return x


def apply_gain_xcore(sample, gain):
    """Apply the gain to a sample usign fixed-point math, assumes that gain is in Q_alpha format"""
    acc = 1 << (Q_alpha - 1)
    acc += sample * gain
    y = utils.int32_mult_sat_extract(acc, 1, Q_alpha)
    return y


def limiter_peak_gain_calc(envelope, threshold, slope=None):
    """Calculate the float gain for the current sample"""
    new_gain = threshold / envelope
    new_gain = min(1, new_gain)
    return new_gain


def limiter_peak_gain_calc_xcore(envelope_int, threshold_int, slope=None):
    """Calculate the int gain for the current sample"""
    if threshold_int >= envelope_int:
        new_gain_int = utils.int32(0x7FFFFFFF)
    else:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
    return new_gain_int


def limiter_rms_gain_calc(envelope, threshold, slope=None):
    """Calculate the float gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain.

    """
    new_gain = sqrt(threshold / envelope)
    new_gain = min(1, new_gain)
    return new_gain


def limiter_rms_gain_calc_xcore(envelope_int, threshold_int, slope=None):
    """Calculate the int gain for the current sample

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
    """Calculate the float gain for the current sample

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
    """Calculate the int gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
    # if envelope below threshold, apply unity gain, otherwise scale
    # down
    if slope_f32 > float32(0) and threshold_int < envelope_int:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
        new_gain_int = ((float32(new_gain_int * 2**-31) ** slope_f32) * float32(2**31)).as_int32()
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


def noise_suppressor_gain_calc(envelope, threshold, slope):
    """Calculate the float gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
    # if envelope above threshold, apply unity gain, otherwise scale
    # down
    new_gain = (threshold / envelope) ** slope
    new_gain = min(1, new_gain)
    return new_gain

def noise_suppressor_gain_calc_xcore(envelope_int, threshold_int, slope_f32):
    """Calculate the int gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
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
        new_gain_int = new_gain_int + 2**31
        new_gain_int = utils.int32(new_gain_int >> 32)
        new_gain_int = ((float32(new_gain_int * 2**-31) ** -slope_f32) * float32(2**31)).as_int32()
    else:
        new_gain_int = utils.int32(0x7FFFFFFF)

    return new_gain_int
