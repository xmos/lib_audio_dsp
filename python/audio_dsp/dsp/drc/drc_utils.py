# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from audio_dsp.dsp import utils as utils
import numpy as np
from math import sqrt


FLT_MIN = np.finfo(float).tiny


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
    alpha = min(2 * T / (attack_or_release_time + FLT_MIN), 1.0)
    return alpha


def limiter_peak_gain_calc(envelope, threshold):
    """Calculate the float gain for the current sample"""
    new_gain = threshold / envelope
    new_gain = min(1, new_gain)
    return new_gain


def limiter_peak_gain_calc_int(envelope_int, threshold_int):
    """Calculate the int gain for the current sample"""
    new_gain = float(threshold_int) / float(envelope_int)
    new_gain = min(1.0, new_gain)
    new_gain_int = utils.int32(new_gain * 2**30)
    return new_gain_int


def limiter_peak_gain_calc_xcore(envelope, threshold_f32):
    """Calculate the np.float32 gain for the current sample"""
    new_gain = threshold_f32 / envelope
    new_gain = new_gain if new_gain < np.float32(1) else np.float32(1)
    return new_gain


def limiter_rms_gain_calc(envelope, threshold):
    """Calculate the float gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain.

    """
    new_gain = sqrt(threshold / envelope)
    new_gain = min(1, new_gain)
    return new_gain


def limiter_rms_gain_calc_int(envelope_int, threshold_int):
    """Calculate the int gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain.

    """
    new_gain = sqrt(float(threshold_int) / float(envelope_int))
    new_gain = min(1.0, new_gain)
    new_gain_int = utils.int32(new_gain * 2**30)
    return new_gain_int


def limiter_rms_gain_calc_xcore(envelope, threshold_f32):
    """Calculate the np.float32 gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain.

    """
    # note use np.sqrt to ensure we stay in f32, using math.sqrt
    # will return float!
    new_gain = np.sqrt(threshold_f32 / envelope)
    new_gain = new_gain if new_gain < np.float32(1) else np.float32(1)
    return new_gain


def compressor_rms_gain_calc(envelope, threshold, slope):
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


def compressor_rms_gain_calc_int(envelope_int, threshold_int, slope_f32):
    """Calculate the int gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
    # if envelope below threshold, apply unity gain, otherwise scale
    # down
    new_gain = (np.float32(threshold_int) / np.float32(envelope_int)) ** slope_f32
    new_gain = min(1.0, new_gain)
    new_gain_int = utils.int32(new_gain * 2**30)
    return new_gain_int


def compressor_rms_gain_calc_xcore(envelope, threshold_f32, slope_f32):
    """Calculate the np.float32 gain for the current sample

    Note that as the RMS envelope detector returns x**2, we need to
    sqrt the gain. Slope is used instead of ratio to allow the gain
    calculation to avoid the log domain.

    """
    # if envelope below threshold, apply unity gain, otherwise scale
    # down
    new_gain = (threshold_f32 / envelope) ** slope_f32
    new_gain = new_gain if new_gain < np.float32(1) else np.float32(1)
    return new_gain