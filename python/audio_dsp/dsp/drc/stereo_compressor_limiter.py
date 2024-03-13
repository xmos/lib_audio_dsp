# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from copy import deepcopy
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp.drc import envelope_detector_peak, envelope_detector_rms

import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32

class compressor_limiter_stereo_base(dspg.dsp_block):
    def __init__(self, fs, n_chans, attack_t, release_t,  Q_sig=dspg.Q_SIG):
        assert n_chans == 2, "has to be stereo"
        super().__init__(fs, n_chans, Q_sig)
    
        self.attack_alpha = drcu.alpha_from_time(attack_t, fs)
        self.release_alpha = drcu.alpha_from_time(release_t, fs)
        self.gain = 1

        # These are defined differently for peak and RMS limiters
        self.threshold = None
        self.env_detector = None

        self.attack_alpha_f32 = float32(self.attack_alpha)
        self.release_alpha_f32 = float32(self.release_alpha)
        self.threshold_f32 = None
        self.gain_f32 = float32(1)

        self.attack_alpha_int = utils.int32(round(self.attack_alpha * 2**30))
        self.release_alpha_int = utils.int32(round(self.release_alpha * 2**30))
        self.threshold_int = None
        self.gain_int = 2**30

    def reset_state(self):
        """Reset the envelope detectors to 0 and the gain to 1."""
        self.env_detector.reset_state()
        self.gain = 1
        self.gain_f32 = float32(1)
        self.gain_int = 2**30
    
    def gain_calc(self, envelope):
        """Calculate the float gain for the current sample"""
        raise NotImplementedError

    def gain_calc_int(self, envelope_int):
        """Calculate the int gain for the current sample"""
        raise NotImplementedError

    def gain_calc_xcore(self, envelope):
        """Calculate the float32 gain for the current sample"""
        raise NotImplementedError
    
    def process_channels(self, input_samples, detect_samples):
        """
        Update the envelopes for a signal, then calculate and apply the
        required gain for compression/limiting, using floating point
        maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        # get envelope from envelope detector
        env0 = self.env_detector.process(detect_samples[0], 0)
        env1 = self.env_detector.process(detect_samples[1], 1)
        envelope = np.maximum(env0, env1)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # calculate the gain, this function should be defined by the
        # child class
        new_gain = self.gain_calc(envelope)

        # see if we're attacking or decaying
        if new_gain < self.gain:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.gain = ((1 - alpha) * self.gain) + (alpha * new_gain)

        # apply gain to input
        y = self.gain * input_samples
        return y, new_gain, envelope
    
    def process_channels_int(self, input_samples, detect_samples):
        """
        Update the envelopes for a signal, then calculate and apply the
        required gain for compression/limiting, using int32 fixed point
        maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        samples_int = [int(0)] * len(detect_samples)
        for i in range(len(detect_samples)): samples_int[i] = utils.int32(round(detect_samples[i] * 2**self.Q_sig))

        # get envelope from envelope detector
        env0_int = self.env_detector.process_int(samples_int[0], 0)
        env1_int = self.env_detector.process_int(samples_int[1], 1)
        envelope_int = max(env0_int, env1_int)
        # avoid /0
        envelope_int = max(envelope_int, 1)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain_int = self.gain_calc_int(envelope_int)

        # see if we're attacking or decaying
        if new_gain_int < self.gain_int:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average, VPU mult uses 2**30, otherwise
        # could use 2**31
        self.gain_int = utils.vpu_mult(2**30 - alpha, self.gain_int)
        self.gain_int += utils.vpu_mult(alpha, new_gain_int)

        y = []
        for i in range(len(input_samples)): samples_int[i] = utils.int32(round(input_samples[i] * 2**self.Q_sig))

        for sample_int in samples_int:
            y_uq = utils.vpu_mult(self.gain_int, sample_int)
            y.append(float(y_uq) * 2 **-self.Q_sig)

        return (
            #(float(y) * 2**-self.Q_sig),
            y,
            (float(new_gain_int) * 2**-self.Q_sig),
            (float(envelope_int) * 2**-self.Q_sig),
        )

    def process_channels_xcore(self, input_samples, detect_samples):
        """
        Update the envelopes for a signal, then calculate and apply the
        required gain for compression/limiting, using float32 maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        # quantize
        samples_int = [int(0)] * len(input_samples)
        samples_f32 = [float32(0)] * len(input_samples)
        for i in range(len(input_samples)):
            samples_int[i] = utils.int32(round(input_samples[i] * 2**self.Q_sig))
            sample_q = utils.float_s32(detect_samples[i])
            sample_q = utils.float_s32_use_exp(sample_q, -27)
            samples_f32[i] = float32(float(sample_q))

        # get envelope from envelope detector
        env0 = self.env_detector.process_xcore(samples_f32[0], 0)
        env1 = self.env_detector.process_xcore(samples_f32[1], 1)
        envelope = np.maximum(env0, env1)
        # avoid /0
        if envelope == float32(0):
            envelope = float32(1e-20)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain = self.gain_calc_xcore(envelope)

        # see if we're attacking or decaying
        if new_gain < self.gain_f32:
            alpha = self.attack_alpha_f32
        else:
            alpha = self.release_alpha_f32

        # do exponential moving average
        self.gain_f32 = self.gain_f32 + alpha * (
            new_gain - self.gain_f32
        )

        # apply gain in int32
        y = [0] * len(samples_int)
        this_gain_int = (self.gain_f32 * float32(2**30)).as_int32()
        for i in range(len(samples_int)):
            acc = int(1 << 29)
            acc += this_gain_int * samples_int[i]
            y_uq = utils.int32_mult_sat_extract(acc, 1, 30)

            # quantize before return
            y[i] = float(y_uq) * 2**-self.Q_sig

        return y, float(new_gain), float(envelope)

    def process_frame(self, frame):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels([frame[0][sample], frame[1][sample]])
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]
        return output
    
    def process_frame_xcore(self, frame):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.
        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_xcore only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels_xcore([frame[0][sample], frame[1][sample]])
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output


class limiter_peak_stereo(compressor_limiter_stereo_base):
    def __init__(self, fs, threshold_dB, attack_t, release_t, Q_sig=dspg.Q_SIG):
        n_chans = 2
        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        self.threshold = utils.db2gain(threshold_dB)
        self.threshold_f32 = float32(self.threshold)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_peak(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

    def process_channels(self, samples):
        return super().process_channels(samples, samples)

    def process_channels_int(self, samples):
        return super().process_channels_int(samples, samples)

    def process_channels_xcore(self, samples):
        return super().process_channels_xcore(samples, samples)

    def gain_calc(self, envelope):
        return drcu.limiter_peak_gain_calc(envelope, self.threshold)

    def gain_calc_int(self, envelope_int):
        return drcu.limiter_peak_gain_calc_int(envelope_int, self.threshold_int)

    def gain_calc_xcore(self, envelope):
        return drcu.limiter_peak_gain_calc_xcore(envelope, self.threshold_f32)


class compressor_rms_stereo(compressor_limiter_stereo_base):
    def __init__(self, fs, ratio, threshold_dB, attack_t, release_t, Q_sig=dspg.Q_SIG):
        n_chans = 2
        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        self.threshold = utils.db_pow2gain(threshold_dB)
        self.threshold_f32 = float32(self.threshold)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.ratio = ratio
        self.slope = (1 - 1 / self.ratio) / 2.0
        self.slope_f32 = float32(self.slope)

    def process_channels(self, samples):
        return super().process_channels(samples, samples)

    def process_channels_int(self, samples):
        return super().process_channels_int(samples, samples)

    def process_channels_xcore(self, samples):
        return super().process_channels_xcore(samples, samples)

    def gain_calc(self, envelope):
        """Calculate the float gain for the current sample

        Note that as the RMS envelope detector returns x**2, we need to
        sqrt the gain. Slope is used instead of ratio to allow the gain
        calculation to avoid the log domain.

        """
        return drcu.compressor_rms_gain_calc(envelope, self.threshold, self.slope)

    def gain_calc_int(self, envelope_int):
        """Calculate the int gain for the current sample

        Note that as the RMS envelope detector returns x**2, we need to
        sqrt the gain. Slope is used instead of ratio to allow the gain
        calculation to avoid the log domain.

        """
        return drcu.compressor_rms_gain_calc_int(envelope_int, self.threshold_int, self.slope_f32)

    def gain_calc_xcore(self, envelope):
        """Calculate the float32 gain for the current sample

        Note that as the RMS envelope detector returns x**2, we need to
        sqrt the gain. Slope is used instead of ratio to allow the gain
        calculation to avoid the log domain.

        """
        return drcu.compressor_rms_gain_calc_xcore(envelope, self.threshold_f32, self.slope_f32)


class compressor_rms_sidechain_stereo(compressor_limiter_stereo_base):
    def __init__(self, fs, ratio, threshold_dB, attack_t, release_t, Q_sig=dspg.Q_SIG):
        n_chans = 2
        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        self.threshold = utils.db_pow2gain(threshold_dB)
        self.threshold_f32 = float32(self.threshold)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.ratio = ratio
        self.slope = (1 - 1 / self.ratio) / 2.0
        self.slope_f32 = float32(self.slope)

    def process_frame(self, frame):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels([frame[0][sample], frame[1][sample]],
                                                      [frame[2][sample], frame[3][sample]])
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]
        return output
    
    def process_frame_xcore(self, frame):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.
        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_xcore only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels_xcore([frame[0][sample], frame[1][sample]],
                                                      [frame[2][sample], frame[3][sample]])
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output

    def gain_calc(self, envelope):
        """Calculate the float gain for the current sample

        Note that as the RMS envelope detector returns x**2, we need to
        sqrt the gain. Slope is used instead of ratio to allow the gain
        calculation to avoid the log domain.

        """
        return drcu.compressor_rms_gain_calc(envelope, self.threshold, self.slope)

    def gain_calc_int(self, envelope_int):
        """Calculate the int gain for the current sample

        Note that as the RMS envelope detector returns x**2, we need to
        sqrt the gain. Slope is used instead of ratio to allow the gain
        calculation to avoid the log domain.

        """
        return drcu.compressor_rms_gain_calc_int(envelope_int, self.threshold_int, self.slope_f32)

    def gain_calc_xcore(self, envelope):
        """Calculate the float32 gain for the current sample

        Note that as the RMS envelope detector returns x**2, we need to
        sqrt the gain. Slope is used instead of ratio to allow the gain
        calculation to avoid the log domain.

        """
        return drcu.compressor_rms_gain_calc_xcore(envelope, self.threshold_f32, self.slope_f32)

