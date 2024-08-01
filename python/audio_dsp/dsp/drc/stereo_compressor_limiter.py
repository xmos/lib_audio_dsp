# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for stereo compressors and limiters."""

from copy import deepcopy

import numpy as np

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp.drc import envelope_detector_peak, envelope_detector_rms

import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32


class compressor_limiter_stereo_base(dspg.dsp_block):
    """
    Stereo compressors and limiters work in a similar way to normal
    compressor_limiter_base objects, but they process two channels and
    apply the same gain to both channels.
    """

    def __init__(
        self, fs, n_chans, threshold_db, attack_t, release_t, envelope_detector, Q_sig=dspg.Q_SIG
    ):
        assert n_chans == 2, "has to be stereo"
        super().__init__(fs, n_chans, Q_sig)

        self.Q_alpha = drcu.Q_alpha
        assert self.Q_alpha == 31, "When changing this the reset value will have to be updated"

        self.env_detector_type = envelope_detector.lower()

        # These are defined differently for peak and RMS limiters
        if self.env_detector_type == "peak":
            self.env_detector = envelope_detector_peak(
                fs,
                n_chans=n_chans,
                attack_t=attack_t,
                release_t=release_t,
                Q_sig=self.Q_sig,
            )
        elif self.env_detector_type == "rms":
            self.env_detector = envelope_detector_rms(
                fs,
                n_chans=n_chans,
                attack_t=attack_t,
                release_t=release_t,
                Q_sig=self.Q_sig,
            )
        else:
            raise ValueError(f"unknown envelope detector type: {envelope_detector}")

        # setting attack and release times sets the EWM coeffs in this and
        # the envelope detector
        self.attack_t = attack_t
        self.release_t = release_t

        # threshold_db should be a property of the child class that sets
        # threshold_int and threshold
        self.threshold_db = threshold_db

        # slope only used by compressors, but needs to be set for gain_calc API
        self.slope = None
        self.slope_f32 = None

        # set the gain calculation function handles
        self.gain_calc = None
        self.gain_calc_xcore = None

        # initialise gain states
        self.reset_state()

    @property
    def threshold_db(self):
        """The threshold in decibels. Setting this also updates the fixed and floating point thresholds in linear gain."""
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value):
        self._threshold_db = value
        if self.env_detector_type == "peak":
            self.threshold, self.threshold_int = drcu.calculate_threshold(
                self._threshold_db, self.Q_sig
            )
        elif self.env_detector_type == "rms":
            self.threshold, self.threshold_int = drcu.calculate_threshold(
                self._threshold_db, self.Q_sig, power=True
            )

    @property
    def attack_t(self):
        """The attack time in seconds. Changing this also sets the EWM alpha values."""
        return self._attack_t

    @attack_t.setter
    def attack_t(self, value):
        self._attack_t = value
        # calculate EWM alpha from time constant
        self.attack_alpha, self.attack_alpha_int = drcu.alpha_from_time(self._attack_t, self.fs)
        # update the envelope detector
        self.env_detector.attack_t = self.attack_t

    @property
    def release_t(self):
        """The release time in seconds. Changing this also sets the EWM alpha values."""
        return self._release_t

    @release_t.setter
    def release_t(self, value):
        self._release_t = value
        # calculate EWM alpha from time constant
        self.release_alpha, self.release_alpha_int = drcu.alpha_from_time(self._release_t, self.fs)
        # update the envelope detector
        self.env_detector.release_t = self.release_t

    def reset_state(self):
        """Reset the envelope detectors to 0 and the gain to 1."""
        if self.env_detector:
            self.env_detector.reset_state()
        self.gain = 1
        self.gain_int = 2**31 - 1

    def process_channels(self, input_samples: list[float]):
        """
        Update the envelopes for a signal, then calculate and apply the
        required gain for compression/limiting, using floating point
        maths. The same gain is applied to both stereo channels.

        Take one new pair of samples and return the compressed/limited
        samples.
        Input should be scaled with 0 dB = 1.0.

        """
        # get envelope from envelope detector
        env0 = self.env_detector.process(input_samples[0], 0)
        env1 = self.env_detector.process(input_samples[1], 1)
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

    def process_channels_xcore(self, input_samples: list[float]):
        """
        Update the envelopes for a signal, then calculate and apply the
        required gain for compression/limiting, using int32 fixed point
        maths. The same gain is applied to both stereo channels.

        Take one new pair of samples and return the compressed/limited
        samples.  The float input sample is quantized to int32, and
        returned to float before outputting. Input should be scaled with
        0 dB = 1.0.

        """
        samples_int = [int(0)] * len(input_samples)
        for i in range(len(input_samples)):
            samples_int[i] = utils.float_to_int32(input_samples[i], self.Q_sig)

        # get envelope from envelope detector
        env0_int = self.env_detector.process_xcore(samples_int[0], 0)
        env1_int = self.env_detector.process_xcore(samples_int[1], 1)
        envelope_int = max(env0_int, env1_int)
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

        y = []

        for sample_int in samples_int:
            y_uq = drcu.apply_gain_xcore(sample_int, self.gain_int)
            y.append(utils.int32_to_float(y_uq, self.Q_sig))

        return (
            y,
            utils.int32_to_float(new_gain_int, self.Q_alpha),
            utils.int32_to_float(envelope_int, self.Q_sig),
        )

    def process_frame(self, frame: list[np.ndarray]):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_channels only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels([frame[0][sample], frame[1][sample]])[0]
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]
        return output

    def process_frame_xcore(self, frame: list[np.ndarray]):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.
        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_channel_xcore only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels_xcore([frame[0][sample], frame[1][sample]])[0]
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output


class limiter_peak_stereo(compressor_limiter_stereo_base):
    """
    A stereo limiter based on the peak value of the signal. When the
    peak envelope of either signal channel, exceeds the threshold, the
    amplitudes of both channels are reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to it's original level
    after the envelope is below the threshold.
    """

    def __init__(self, fs, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        n_chans = 2
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, "peak", Q_sig)

        # set the gain calculation function handles
        self.gain_calc = drcu.limiter_peak_gain_calc
        self.gain_calc_xcore = drcu.limiter_peak_gain_calc_xcore


class compressor_rms_stereo(compressor_limiter_stereo_base):
    """
    A stereo compressor based on the RMS value of the signal. When the
    RMS envelope of the either signal channel exceeds the threshold, the
    amplitudes of both channels are reduced by the compression ratio.

    The threshold sets the value above which compression occurs. The
    ratio sets how much the signal is compressed. A ratio of 1 results
    in no compression, while a ratio of infinity results in the same
    behaviour as a limiter. The attack time sets how fast the compressor
    starts compressing. The release time sets how long the signal takes
    to ramp up to it's original level after the envelope is below the
    threshold.
    """

    def __init__(self, fs, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        n_chans = 2
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, "rms", Q_sig)

        # property calculates the slopes as well
        self.ratio = ratio

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore

    @property
    def ratio(self):
        """Compression gain ratio applied when the signal is above the threshold."""
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value
        self.slope, self.slope_f32 = drcu.compressor_slope_from_ratio(self.ratio)
