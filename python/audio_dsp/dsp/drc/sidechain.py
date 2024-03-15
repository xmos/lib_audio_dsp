# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from copy import deepcopy

import numpy as np


from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32

from audio_dsp.dsp.drc.drc import compressor_limiter_base, envelope_detector_rms
from audio_dsp.dsp.drc.stereo_compressor_limiter import compressor_limiter_stereo_base


class compressor_rms_sidechain_mono(compressor_limiter_base):
    """
    A sidechain compressor based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced by the compression ratio.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the comressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to it's original level after the envelope is below the
    threshold.

    Parameters
    ----------
    ratio : float
        Compression gain ratio applied when the signal is above the
        threshold

    Attributes
    ----------
    env_detector : envelope_detector_rms
        Nested RMS envelope detector used to calculate the envelope of
        the signal.
    ratio : float
        Compression gain ratio applied when the signal is above the
        threshold.
    slope : float
        The slope factor of the compressor, defined as
        `slope = (1 - 1/ratio)`.
    slope : float32
        The slope factor of the compressor, used for int32 to float32
        processing.
    threshold : float
        Value above which compression occurs for floating point
        processing.
    threshold_f32 : float32
        Value above which compression occurs for floating point
        processing.
    threshold_int : int
        Value above which compression occurs for int32 fixed point
        processing.

    """

    def __init__(
        self, fs, ratio, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG
    ):
        super().__init__(fs, 1, attack_t, release_t, delay, Q_sig)

        # note rms comes as x**2, so use db_pow
        self.threshold = utils.db_pow2gain(threshold_db)
        self.threshold_f32 = float32(self.threshold)
        self.threshold_int = utils.int32(self.threshold * 2**self.Q_sig)
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=1,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.ratio = ratio
        self.slope = (1 - 1 / self.ratio) / 2.0
        self.slope_f32 = float32(self.slope)

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_int = drcu.compressor_rms_gain_calc_int
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore

    def process(self, input_sample, detect_sample, channel=0):
        """
        Update the envelope for the detection signal, then calculate and
        apply the required gain for compression/limiting, and apply to
        the input signal using floating point maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        # get envelope from envelope detector
        envelope = self.env_detector.process(detect_sample, channel)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # calculate the gain, this function should be defined by the
        # child class
        new_gain = self.gain_calc(envelope, self.threshold, self.slope)

        # see if we're attacking or decaying
        if new_gain < self.gain[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.gain[channel] = ((1 - alpha) * self.gain[channel]) + (alpha * new_gain)

        # apply gain to input
        y = self.gain[channel] * input_sample
        return y, new_gain, envelope

    def process_int(self, input_sample, detect_sample, channel=0):
        raise NotImplementedError

    def process_xcore(self, input_sample, detect_sample, channel=0):
        """
        Update the envelope for the detection signal, then calculate and
        apply the required gain for compression/limiting, and apply to
        the input signal using float32 maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0dB = 1.0.

        """
        # quantize
        sample_int = utils.int32(round(input_sample * 2**self.Q_sig))
        detect_sample = utils.float_s32(detect_sample)
        detect_sample = utils.float_s32_use_exp(detect_sample, -27)
        detect_sample = float32(float(detect_sample))

        # get envelope from envelope detector
        envelope = self.env_detector.process_xcore(detect_sample, channel)
        # avoid /0
        if envelope == float32(0):
            envelope = float32(1e-20)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain = self.gain_calc_xcore(envelope, self.threshold_f32, self.slope_f32)

        # see if we're attacking or decaying
        if new_gain < self.gain_f32[channel]:
            alpha = self.attack_alpha_f32
        else:
            alpha = self.release_alpha_f32

        # do exponential moving average
        self.gain_f32[channel] = self.gain_f32[channel] + alpha * (
            new_gain - self.gain_f32[channel]
        )

        # apply gain in int32
        this_gain_int = (self.gain_f32[channel] * float32(2**30)).as_int32()
        acc = int(1 << 29)
        acc += this_gain_int * sample_int
        y = utils.int32_mult_sat_extract(acc, 1, 30)

        # quantize before return
        y = float(y) * 2**-self.Q_sig

        return y, float(new_gain), float(envelope)

    def process_frame(self, frame):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process only take the first output.

        """
        assert len(frame) == 2
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process(frame[0][sample], frame[1][sample])[0]


        return [output]

    def process_frame_xcore(self, frame):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.
        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_xcore only take the first output.

        """
        assert len(frame) == 2
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process_xcore(frame[0][sample], frame[1][sample])[0]

        return [output]



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

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_int = drcu.compressor_rms_gain_calc_int
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore

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
        new_gain = self.gain_calc(envelope, self.threshold, self.slope)

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
        new_gain_int = self.gain_calc_int(envelope_int, self.threshold_int, self.slope_f32)

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
        new_gain = self.gain_calc_xcore(envelope, self.threshold_f32, self.slope_f32)

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
        assert len(frame) == 4, "has to be dual stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame[0:2])
        for sample in range(frame_size):
            out_samples = self.process_channels([frame[0][sample], frame[1][sample]],
                                                      [frame[2][sample], frame[3][sample]])[0]
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
        assert len(frame) == 4, "has to be dual stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame[0:2])
        for sample in range(frame_size):
            out_samples = self.process_channels_xcore([frame[0][sample], frame[1][sample]],
                                                      [frame[2][sample], frame[3][sample]])[0]
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output