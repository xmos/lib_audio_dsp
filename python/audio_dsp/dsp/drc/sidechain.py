# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The sidechain compressor DSP blocks."""

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
    A mono sidechain compressor based on the RMS value of the signal.
    When the RMS envelope of the signal exceeds the threshold, the
    signal amplitude is reduced by the compression ratio.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the compressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to it's original level after the envelope is below the
    threshold.

    Parameters
    ----------
    ratio : float
        Compression gain ratio applied when the signal is above the
        threshold
    threshold_db : float
        Threshold in decibels above which compression occurs.

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
    slope_f32 : float32
        The slope factor of the compressor, used for int32 to float32
        processing.
    threshold : float
        Value above which compression occurs for floating point
        processing.
    threshold_int : int
        Value above which compression occurs for int32 fixed point
        processing.

    """

    def __init__(self, fs, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, 1, threshold_db, attack_t, release_t, Q_sig)

        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=1,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.slope = (1 - 1 / ratio) / 2.0
        self.slope_f32 = float32(self.slope)

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore

    @property
    def threshold_db(self):
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value):
        self._threshold_db = value
        # note rms comes as x**2, so use db_pow
        self.threshold, self.threshold_int = drcu.calculate_threshold(
            self.threshold_db, self.Q_sig, power=True
        )


    def reset_state(self):
        """Reset the envelope detectors to 0 and the gain to 1."""
        if self.env_detector:
            self.env_detector.reset_state()
        self.gain = 1
        self.gain_int = 2**31 - 1

    def process(self, input_sample: float, detect_sample: float):  # type: ignore : overloading base class
        """
        Update the envelope for the detection signal, then calculate and
        apply the required gain for compression/limiting, and apply to
        the input signal using floating point maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0 dB = 1.0.

        Parameters
        ----------
        input_sample : float
            The input sample to be compressed.
        detect_sample : float
            The sample used by the envelope detector to determine the
            amount of compression to apply to the input_sample.
        """
        # get envelope from envelope detector
        envelope = self.env_detector.process(detect_sample)
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
        y = self.gain * input_sample
        return y, new_gain, envelope

    def process_xcore(self, input_sample: float, detect_sample: float):  # type: ignore : overloading base class
        """
        Update the envelope for the detection signal, then calculate and
        apply the required gain for compression/limiting, and apply to
        the input signal using int32 maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0 dB = 1.0.

        Parameters
        ----------
        input_sample : float
            The input sample to be compressed.
        detect_sample : float
            The sample used by the envelope detector to determine the
            amount of compression to apply to the input_sample.

        """
        # quantize
        sample_int = utils.float_to_int32(input_sample, self.Q_sig)
        detect_sample_int = utils.float_to_int32(detect_sample, self.Q_sig)

        # get envelope from envelope detector
        envelope_int = self.env_detector.process_xcore(detect_sample_int)
        # avoid /0
        envelope_int = max(envelope_int, 1)
        assert isinstance(envelope_int, int)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain_int = self.gain_calc_xcore(envelope_int, self.threshold_int, self.slope_f32)

        # see if we're attacking or decaying
        if new_gain_int < self.gain_int:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        self.gain_int = drcu.calc_ema_xcore(self.gain_int, new_gain_int, alpha)

        # apply gain
        y = drcu.apply_gain_xcore(sample_int, self.gain_int)

        return (
            utils.int32_to_float(y, self.Q_sig),
            utils.int32_to_float(new_gain_int, self.Q_alpha),
            utils.int32_to_float(envelope_int, self.Q_sig),
        )

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
    """
    A stereo sidechain compressor based on the RMS value of the signal.
    When the RMS envelope of the signal exceeds the threshold, the
    signal amplitude is reduced by the compression ratio. The same
    compression is applied to both channels, using the highest
    individual channel envelope.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the compressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to it's original level after the envelope is below the
    threshold.

    Parameters
    ----------
    ratio : float
        Compression gain ratio applied when the signal is above the
        threshold
    threshold_db : float
        Threshold in decibels above which compression occurs.

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
    slope_f32 : float32
        The slope factor of the compressor, used for int32 to float32
        processing.
    threshold : float
        Value above which compression occurs for floating point
        processing.
    threshold_int : int
        Value above which compression occurs for int32 fixed point
        processing.

    """

    def __init__(self, fs, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        n_chans = 2
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=self.Q_sig,
        )

        self.slope = (1 - 1 / ratio) / 2.0
        self.slope_f32 = float32(self.slope)

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore

    @property
    def threshold_db(self):
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value):
        self._threshold_db = value
        # note rms comes as x**2, so use db_pow
        self.threshold, self.threshold_int = drcu.calculate_threshold(
            self.threshold_db, self.Q_sig, power=True
        )


    def process_channels(self, input_samples: list[float], detect_samples: list[float]):  # type: ignore : override base class
        """
        Update the envelopes for a detection signal, then calculate and
        apply the required gain for compression/limiting to the input,
        using floating point maths. The same gain is applied to both
        stereo channels.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0 dB = 1.0.

        Parameters
        ----------
        input_samples : list[float]
            List of input samples to be compressed.
        detect_samples : list[float]
            List of samples used by the envelope detector to determine the
            amount of compression to apply to the input_sample.

        """
        # get envelope from envelope detector
        env0 = self.env_detector.process(detect_samples[0], 0)
        env1 = self.env_detector.process(detect_samples[1], 1)
        envelope = np.maximum(env0, env1)
        # avoid /0
        envelope = np.maximum(envelope, np.finfo(float).tiny)

        # calculate the gain, this function should be defined by the
        # child class
        new_gain = self.gain_calc(envelope, self.threshold, self.slope)  # type: ignore : base inits to None

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

    def process_channels_xcore(  # type: ignore : override base class
        self, input_samples: list[float], detect_samples: list[float]
    ):
        """
        Update the envelopes for a detection signal, then calculate and
        apply the required gain for compression/limiting to the input,
        using int32 fixed point maths. The same gain is applied to both
        stereo channels.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0 dB = 1.0.

        Parameters
        ----------
        input_samples : list[float]
            List of input samples to be compressed.
        detect_samples : list[float]
            List of samples used by the envelope detector to determine the
            amount of compression to apply to the input_sample.

        """
        # quantize
        samples_int = [int(0)] * len(input_samples)
        detect_samples_int = [int(0)] * len(input_samples)
        for i in range(len(input_samples)):
            samples_int[i] = utils.float_to_int32(input_samples[i], self.Q_sig)
            detect_samples_int[i] = utils.float_to_int32(detect_samples[i], self.Q_sig)

        # get envelope from envelope detector
        env0 = self.env_detector.process_xcore(detect_samples_int[0], 0)
        env1 = self.env_detector.process_xcore(detect_samples_int[1], 1)
        envelope_int = np.maximum(env0, env1)
        # avoid /0
        envelope_int = max(envelope_int, 1)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain_int = self.gain_calc_xcore(envelope_int, self.threshold_int, self.slope_f32)  # type: ignore : base inits to None

        # see if we're attacking or decaying
        if new_gain_int < self.gain_int:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        self.gain_int = drcu.calc_ema_xcore(self.gain_int, new_gain_int, alpha)

        y = [0.0] * self.n_chans
        # apply gain in int32
        for i in range(len(input_samples)):
            y_uq = drcu.apply_gain_xcore(samples_int[i], self.gain_int)
            y[i] = utils.int32_to_float(y_uq, self.Q_sig)

        return (
            y,
            utils.int32_to_float(new_gain_int, self.Q_alpha),
            utils.int32_to_float(envelope_int, self.Q_sig),
        )

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
            out_samples = self.process_channels(
                [frame[0][sample], frame[1][sample]],
                [frame[2][sample], frame[3][sample]],
            )[0]
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
            out_samples = self.process_channels_xcore(
                [frame[0][sample], frame[1][sample]],
                [frame[2][sample], frame[3][sample]],
            )[0]
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output
