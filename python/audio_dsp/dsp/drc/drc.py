# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The dynamic range control (DRC) DSP blocks."""

from copy import deepcopy

import numpy as np

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.types import float32


FLT_MIN = np.finfo(float).tiny


class envelope_detector_peak(dspg.dsp_block):
    """
    Envelope detector that follows the absolute peak value of a signal.

    The attack time sets how fast the envelope detector ramps up. The
    release time sets how fast the envelope detector ramps down.

    Parameters
    ----------
    attack_t : float
        Attack time of the envelope detector in seconds. This cannot be
        faster than the length of 2 samples, and saturates to that
        value. Exceptionally large attack times may converge to zero.
    release_t: float
        Release time of the envelope detector in seconds. This cannot
        be faster than the length of 2 samples, and saturates to that
        value. Exceptionally large release times may converge to zero.

    Attributes
    ----------
    attack_t : float
    release_t : float
    attack_alpha : float
        Attack time parameter used for exponential moving average in
        floating point processing.
    release_alpha : float
        Release time parameter used for exponential moving average in
        floating point processing.
    envelope : list[float]
        Current envelope value for each channel for floating point
        processing.
    attack_alpha_int : int
        attack_alpha in 32-bit int format.
    release_alpha_int : int
        release_alpha in 32-bit int format.
    envelope_int : list[int]
        current envelope value for each channel in 32-bit int format.

    """

    def __init__(self, fs, n_chans, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.attack_t = attack_t
        self.release_t = release_t

        # initialise envelope state
        self.reset_state()

    @property
    def attack_t(self):
        """The attack time of the compressor/limiter in seconds;
        changing this property also sets the EWM alpha values for fixed
        and floating point implementations.
        """
        return self._attack_t

    @attack_t.setter
    def attack_t(self, value):
        self._attack_t = value
        # calculate EWM alpha from time constant
        self.attack_alpha, self.attack_alpha_int = drcu.alpha_from_time(self._attack_t, self.fs)

    @property
    def release_t(self):
        """The release time of the compressor/limiter in seconds;
        changing this property also sets the EWM alpha values for fixed
        and floating point implementations.
        """
        return self._release_t

    @release_t.setter
    def release_t(self, value):
        self._release_t = value
        # calculate EWM alpha from time constant
        self.release_alpha, self.release_alpha_int = drcu.alpha_from_time(self._release_t, self.fs)

    def reset_state(self):
        """Reset the envelope to zero."""
        self.envelope = [0.0] * self.n_chans
        self.envelope_int = [utils.int32(0)] * self.n_chans

    def process(self, sample, channel=0):
        """
        Update the peak envelope for a signal, using floating point
        maths.

        Take one new sample and return the updated envelope. Input
        should be scaled with 0 dB = 1.0.

        """
        if isinstance(sample, list) or isinstance(sample, np.ndarray):
            sample = sample[channel]

        sample_mag = abs(sample)

        # see if we're attacking or decaying
        if sample_mag > self.envelope[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1 - alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_xcore(self, sample, channel=0):
        """
        Update the peak envelope for a signal, using int32 fixed point
        maths.

        Take 1 new sample and return the updated envelope. If the input
        is int, return a int, otherwise expect float input
        and return float output.

        """
        if isinstance(sample, float):
            sample_int = utils.float_to_int32(sample, self.Q_sig)
        elif (isinstance(sample, list) or isinstance(sample, np.ndarray)) and isinstance(
            sample[0], int
        ):
            sample_int = sample[channel]
        elif isinstance(sample, int):
            sample_int = sample
        else:
            raise TypeError("input must be float or int")

        sample_mag = abs(sample_int)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_int[channel]:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        self.envelope_int[channel] = drcu.calc_ema_xcore(
            self.envelope_int[channel], sample_mag, alpha
        )

        if isinstance(sample, float):
            return utils.int32_to_float(self.envelope_int[channel], self.Q_sig)
        else:
            return self.envelope_int[channel]


class envelope_detector_rms(envelope_detector_peak):
    """
    Envelope detector that follows the RMS value of a signal.

    Note this returns the mean² value, there is no need to do the
    sqrt() as if the output is converted to dB, 10log10() can be taken
    instead of 20log10().

    The attack time sets how fast the envelope detector ramps up. The
    release time sets how fast the envelope detector ramps down.

    """

    def process(self, sample, channel=0):
        """
        Update the RMS envelope for a signal, using floating point
        maths.

        Take one new sample and return the updated envelope. Input
        should be scaled with 0 dB = 1.0.

        Note this returns the mean² value, there is no need to do the
        sqrt() as if the output is converted to dB, 10log10() can be
        taken instead of 20log10().

        """
        # for rms use power
        sample_mag = sample**2

        # see if we're attacking or decaying
        if sample_mag > self.envelope[channel]:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        # do exponential moving average
        self.envelope[channel] = ((1 - alpha) * self.envelope[channel]) + (alpha * sample_mag)

        return self.envelope[channel]

    def process_xcore(self, sample, channel=0):
        """
        Update the RMS envelope for a signal, using int32 fixed point
        maths.

        Take one new sample and return the updated envelope. Input
        should be scaled with 0 dB = 1.0.

        Note this returns the mean² value, there is no need to do the
        sqrt() as if the output is converted to dB, 10log10() can be
        taken instead of 20log10().

        """
        if isinstance(sample, float):
            sample_int = utils.float_to_int32(sample, self.Q_sig)
        else:
            sample_int = sample

        acc = int(1 << (self.Q_sig - 1))
        acc += sample_int * sample_int
        sample_mag = utils.int32_mult_sat_extract(acc, 1, self.Q_sig)

        # see if we're attacking or decaying
        if sample_mag > self.envelope_int[channel]:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        self.envelope_int[channel] = drcu.calc_ema_xcore(
            self.envelope_int[channel], sample_mag, alpha
        )

        # if we got floats, return floats, otherwise return ints
        if isinstance(sample, float):
            return utils.int32_to_float(self.envelope_int[channel], self.Q_sig)
        else:
            return self.envelope_int[channel]


class clipper(dspg.dsp_block):
    """
    A simple clipper that limits the signal to a specified threshold.

    Parameters
    ----------
    threshold_db : float
        Threshold above which clipping occurs in dB.

    Attributes
    ----------
    threshold_db : float
    threshold : float
        Value above which clipping occurs for floating point processing.
    threshold_int : int
        Value above which clipping occurs for int32 fixed point
        processing.

    """

    def __init__(self, fs, n_chans, threshold_db, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.threshold_db = threshold_db

    @property
    def threshold_db(self):
        """The threshold in decibels; changing this property also
        updates the fixed and floating point thresholds in linear gain.
        """
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value):
        self._threshold_db = value
        self.threshold, self.threshold_int = drcu.calculate_threshold(
            self._threshold_db, self.Q_sig
        )

    def process(self, sample, channel=0):
        """
        Take one new sample and return the clipped sample, using
        floating point maths.
        Input should be scaled with 0 dB = 1.0.
        """
        if sample > self.threshold:
            return self.threshold
        elif sample < -self.threshold:
            return -self.threshold
        else:
            return sample

    def process_xcore(self, sample, channel=0):
        """
        Take one new sample and return the clipped sample, using int32
        fixed point maths.
        Input should be scaled with 0 dB = 1.0.
        """
        # convert to int
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        # do the clipping
        if sample_int > self.threshold_int:
            sample_int = self.threshold_int
        elif sample_int < -self.threshold_int:
            sample_int = -self.threshold_int

        return utils.int32_to_float(sample_int, self.Q_sig)


class compressor_limiter_base(dspg.dsp_block):
    """
    A base class shared by compressor and limiter objects.

    The compressor and limiter have very similar structures, with
    differences in the gain calculation. All the shared code and
    parameters are calculated in this base class.

    Parameters
    ----------
    n_chans : int
        number of parallel channels the compressor/limiter runs on. The
        channels are limited/compressed separately, only the constant
        parameters are shared.
    attack_t : float
        Attack time of the compressor/limiter in seconds. This cannot be
        faster than the length of 2 samples, and saturates to that
        value. Exceptionally large attack times may converge to zero.
    release_t: float
        Release time of the compressor/limiter in seconds. This cannot
        be faster than the length of 2 samples, and saturates to that
        value. Exceptionally large release times may converge to zero.


    Attributes
    ----------
    attack_t : float
    release_t : float
    threshold : float
        Value above which compression/limiting occurs for floating point
        processing.
    gain : list[float]
        Current gain to be applied to the signal for each channel for
        floating point processing.
    attack_alpha : float
        Attack time parameter used for exponential moving average in
        floating point processing.
    release_alpha : float
        Release time parameter used for exponential moving average in
        floating point processing.
    threshold_int : int
        Value above which compression/limiting occurs for int32 fixed
        point processing.
    gain_int : list[int]
        Current gain to be applied to the signal for each channel for
        int32 fixed point processing.
    attack_alpha_int : int
        attack_alpha in 32-bit int format.
    release_alpha_int : int
        release_alpha in 32-bit int format.
    gain_calc : function
        function pointer to floating point gain calculation function.
    gain_calc_int : function
        function pointer to fixed point gain calculation function.
    """

    # Limiter after Zolzer's DAFX & Guy McNally's "Dynamic Range Control
    # of Digital Audio Signals"
    def __init__(self, fs, n_chans, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        self.Q_alpha = drcu.Q_alpha
        assert self.Q_alpha == 31, "When changing this the reset value will have to be updated"

        # setting attack and release times sets the EWM coeffs in this and
        # the envelope detector
        self.attack_t = attack_t
        self.release_t = release_t

        # slope only used by compressors, but needs to be set for gain_calc API
        self.slope = None
        self.slope_f32 = None

        # initialise gain states
        self.reset_state()

        # set the gain calculation function handles
        self.gain_calc = None
        self.gain_calc_xcore = None

    @property
    def attack_t(self):
        """The attack time of the compressor/limiter in seconds;
        changing this property also sets the EWM alpha values for fixed
        and floating point implementations.
        """
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
        """The release time of the compressor/limiter in seconds;
        changing this property also sets the EWM alpha values for fixed
        and floating point implementations.
        """
        return self._release_t

    @release_t.setter
    def release_t(self, value):
        self._release_t = value
        # calculate EWM alpha from time constant
        self.release_alpha, self.release_alpha_int = drcu.alpha_from_time(self._release_t, self.fs)
        # update the envelope detector
        self.env_detector.release_t = self.release_t

    def reset_state(self):
        """Reset the envelope detector to 0 and the gain to 1."""
        if self.env_detector:
            self.env_detector.reset_state()
        self.gain = [1] * self.n_chans
        self.gain_int = [2**31 - 1] * self.n_chans

    def get_gain_curve(self, max_gain=dspg.HEADROOM_DB, min_gain=-96):
        """Get the compression gain curve for the float implementation,
        showing the relationship between the input and output gain.
        """
        in_gains_db = np.linspace(min_gain, max_gain, 1000)
        gains_lin = utils.db2gain(in_gains_db)

        if isinstance(self.env_detector, envelope_detector_rms):
            # if RMS compressor, we need to use gains_lin**2 into the
            # gain calc
            gains_lin = gains_lin**2

        out_gains = np.zeros_like(gains_lin)

        for n in range(len(out_gains)):
            out_gains[n] = self.gain_calc(gains_lin[n], self.threshold, self.slope)  # pyright: ignore : function handles inits to None

        out_gains_db = utils.db(out_gains) + in_gains_db

        return in_gains_db, out_gains_db

    def get_gain_curve_int(self, max_gain=dspg.HEADROOM_DB, min_gain=-96):
        """Get the compression gain curve for the int32 implementation,
        showing the relationship between the input and output gain.
        """
        in_gains_db = np.linspace(min_gain, max_gain, 1000)
        gains_lin = utils.db2gain(in_gains_db)

        if isinstance(self.env_detector, envelope_detector_rms):
            # if RMS compressor, we need to use gains_lin**2 into the
            # gain calc
            gains_lin = gains_lin**2

        out_gains = np.zeros_like(gains_lin)

        for n in range(len(out_gains)):
            # NOTE, if RMS compressor, we need to use gains_lin**2
            out_gains[n] = self.gain_calc_xcore(
                utils.int32(round(gains_lin[n] * 2**self.Q_sig)),
                self.threshold_int,
                self.slope_f32,
            )  # pyright: ignore : base inits to None
            out_gains[n] = float(out_gains[n]) * 2**-31

        out_gains_db = utils.db(out_gains) + in_gains_db

        return in_gains_db, out_gains_db

    def process(self, sample, channel=0):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for compression/limiting, using floating point
        maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0 dB = 1.0.

        """
        # get envelope from envelope detector
        envelope = self.env_detector.process(sample, channel)
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
        y = self.gain[channel] * sample
        return y, new_gain, envelope

    def process_xcore(self, sample: float, channel=0, return_int=False):  # pyright: ignore : overload base class
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for compression/limiting, using int32 fixed point
        maths.

        Take one new sample and return the compressed/limited sample.
        Input should be scaled with 0 dB = 1.0.

        Parameters
        ----------
        return_int : bool
            If True, return int scaled values. If False, rescale to
            floating point, scaled by ``2**-self.Q_sig``.

        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)
        # get envelope from envelope detector
        envelope_int = self.env_detector.process_xcore(sample_int, channel)  # type: ignore : base inits to None
        # avoid /0
        envelope_int = max(envelope_int, 1)

        # if envelope below threshold, apply unity gain, otherwise scale
        # down
        new_gain_int = self.gain_calc_xcore(envelope_int, self.threshold_int, self.slope_f32)  # pyright: ignore : base inits to None

        # see if we're attacking or decaying
        if new_gain_int < self.gain_int[channel]:
            alpha = self.attack_alpha_int
        else:
            alpha = self.release_alpha_int

        # do exponential moving average
        self.gain_int[channel] = drcu.calc_ema_xcore(self.gain_int[channel], new_gain_int, alpha)

        # apply gain
        y = drcu.apply_gain_xcore(sample_int, self.gain_int[channel])

        if return_int:
            return y, new_gain_int, envelope_int
        else:
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
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process(this_chan[sample], channel=chan)[0]

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
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_xcore(this_chan[sample], channel=chan)[0]

        return output


class peak_compressor_limiter_base(compressor_limiter_base):
    """
    A compressor/limiter with a peak envelope detector.

    Attributes
    ----------
    threshold_db : float
    env_detector : envelope_detector_peak
        Nested peak envelope detector used to calculate the envelope of
        the signal.
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        self.env_detector = envelope_detector_peak(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )

        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        # threshold_db should be a property of the child class that sets
        # threshold_int and threshold
        self.threshold_db = threshold_db

    @property
    def threshold_db(self):
        """The threshold in decibels; changing this property also
        updates the fixed and floating point thresholds in linear
        gain.
        """
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value):
        self._threshold_db = value
        self.threshold, self.threshold_int = drcu.calculate_threshold(
            self._threshold_db, self.Q_sig
        )


class rms_compressor_limiter_base(compressor_limiter_base):
    """
    A compressor/limiter with an RMS envelope detector.

    Note the threshold is saved in the power domain, as the RMS envelope
    detector returns x².

    Attributes
    ----------
    env_detector : envelope_detector_rms
        Nested RMS envelope detector used to calculate the envelope of
        the signal.
    threshold : float
        Value above which compression/limiting occurs for floating point
        processing. Note the threshold is saved in the power domain, as
        the RMS envelope detector returns x².
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        self.env_detector = envelope_detector_rms(
            fs,
            n_chans=n_chans,
            attack_t=attack_t,
            release_t=release_t,
            Q_sig=Q_sig,
        )

        super().__init__(fs, n_chans, attack_t, release_t, Q_sig)

        # threshold_db should be a property of the child class that sets
        # threshold_int and threshold
        self.threshold_db = threshold_db

    @property
    def threshold_db(self):
        """The threshold in decibels; changing this property also
        updates the fixed and floating point thresholds in linear
        gain.
        """
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value):
        self._threshold_db = value
        self.threshold, self.threshold_int = drcu.calculate_threshold(
            self._threshold_db, self.Q_sig, power=True
        )


class limiter_peak(peak_compressor_limiter_base):
    """
    A limiter based on the peak value of the signal. When the peak
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to its original level
    after the envelope is below the threshold.
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        # set the gain calculation function handles
        self.gain_calc = drcu.limiter_peak_gain_calc
        self.gain_calc_xcore = drcu.limiter_peak_gain_calc_xcore


class limiter_rms(rms_compressor_limiter_base):
    """
    A limiter based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced.

    The threshold set the value above which limiting occurs. The attack
    time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to its original level
    after the envelope is below the threshold.
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        # set the gain calculation function handles
        self.gain_calc = drcu.limiter_rms_gain_calc
        self.gain_calc_xcore = drcu.limiter_rms_gain_calc_xcore


class hard_limiter_peak(limiter_peak):
    """
    A limiter based on the peak value of the signal. When the peak
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced. If the signal still exceeds the threshold, it is
    clipped.

    The threshold set the value above which limiting/clipping occurs.
    The attack time sets how fast the limiter starts limiting. The
    release time sets how long the signal takes to ramp up to it's
    original level after the envelope is below the threshold.
    """

    def process(self, sample, channel=0):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for limiting, using floating point maths. If the
        output signal exceeds the threshold, clip it to the threshold.

        Take one new sample and return the limited sample.
        Input should be scaled with 0 dB = 1.0.

        """
        y, new_gain, envelope = super().process(sample, channel)

        # hard clip if above threshold
        if y > self.threshold:
            y = self.threshold
        elif y < -self.threshold:
            y = -self.threshold
        return y, new_gain, envelope

    def process_xcore(self, sample, channel=0, return_int=False):
        """
        Update the envelope for a signal, then calculate and apply the
        required gain for limiting, using int32 fixed point maths. If
        the output signal exceeds the threshold, clip it to the
        threshold.

        Take one new sample and return the limited sample.
        Input should be scaled with 0 dB = 1.0.

        """
        y, new_gain_int, envelope_int = super().process_xcore(sample, channel, return_int=True)

        assert isinstance(y, int)
        assert isinstance(new_gain_int, int)
        assert isinstance(envelope_int, int)

        # hard clip if above threshold
        if y > self.threshold_int:
            y = self.threshold_int
        elif y < -self.threshold_int:
            y = -self.threshold_int

        if return_int:
            return y, new_gain_int, envelope_int
        else:
            return (
                utils.int32_to_float(y, self.Q_sig),
                utils.int32_to_float(new_gain_int, self.Q_alpha),
                utils.int32_to_float(envelope_int, self.Q_sig),
            )


class lookahead_limiter_peak(peak_compressor_limiter_base):
    """Not implemented. Peak limiter with built in delay for avoiding
    clipping.
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        self.delay = np.ceil(attack_t * fs)
        self.delay_line = np.zeros(self.delay)
        raise NotImplementedError

    def process(self, sample, channel=0):
        """Not implemented."""
        raise NotImplementedError

    def process_xcore(self, sample, channel=0, return_int=False):
        """Not implemented."""
        raise NotImplementedError


class lookahead_limiter_rms(rms_compressor_limiter_base):
    """Not implemented. RMS limiter with built in delay for avoiding
    clipping.
    """

    def __init__(self, fs, n_chans, threshold_db, attack_t, release_t, delay, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        self.delay = np.ceil(attack_t * fs)
        self.delay_line = np.zeros(self.delay)
        raise NotImplementedError

    def process(self, sample, channel=0):
        """Not implemented."""
        raise NotImplementedError

    def process_xcore(self, sample, channel=0, return_int=False):
        """Not implemented."""
        raise NotImplementedError


# TODO lookahead limiters and compressors
# TODO add soft limiter
# TODO add peak compressors
# TODO add soft knee compressors
# TODO add lookup compressors w/ some magic interface


class compressor_rms(rms_compressor_limiter_base):
    """
    A compressor based on the RMS value of the signal. When the RMS
    envelope of the signal exceeds the threshold, the signal amplitude
    is reduced by the compression ratio.

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

    Attributes
    ----------
    ratio : float
    slope : float
        The slope factor of the compressor, defined as
        `slope = (1 - 1/ratio) / 2`.
    slope_f32 : float32
        The slope factor of the compressor, used for int32 to float32
        processing.

    """

    def __init__(self, fs, n_chans, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        # property calculates the slopes as well
        self.ratio = ratio

        # set the gain calculation function handles
        self.gain_calc = drcu.compressor_rms_gain_calc
        self.gain_calc_xcore = drcu.compressor_rms_gain_calc_xcore

    @property
    def ratio(self):
        """Compression gain ratio applied when the signal is above the
        threshold; changing this property also updates the slope used in
        the fixed and floating point implementation.
        """
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value
        self.slope, self.slope_f32 = drcu.rms_compressor_slope_from_ratio(self.ratio)


class compressor_rms_softknee(rms_compressor_limiter_base):
    """
    A soft knee compressor based on the RMS value of the signal. When
    the RMS envelope of the signal exceeds the threshold, the signal
    amplitude is reduced by the compression ratio. A smoothed fit is
    used around the knee to reduce artifacts.

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

    Attributes
    ----------
    ratio : float
    slope : float
        The slope factor of the compressor, defined as
        `slope = (1 - 1/ratio) / 2`.
    slope_f32 : np.float32
        The slope factor of the compressor, used for int32 to float32
        processing.
    w : float
        The width over which the soft knee extends.

    References
    ----------
    [1] Giannoulis, D., Massberg, M., & Reiss, J. D. (2012). Digital
    Dynamic Range Compressor Design - A Tutorial and Analysis. Journal of
    Audio Engineering Society, 60(6), 399-408.
    https://www.aes.org/e-lib/browse.cfm?elib=16354
    """

    def __init__(self, fs, n_chans, ratio, threshold_db, attack_t, release_t, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, threshold_db, attack_t, release_t, Q_sig)

        self.ratio = ratio

        # this is a bit of a bodge, as the soft knee compressor needs
        # more inputs
        self.gain_calc = self.compressor_rms_softknee_gain_calc
        self.gain_calc_xcore = self.compressor_rms_softknee_gain_calc_xcore

    @property
    def ratio(self):
        """Compression gain ratio applied when the signal is above the
        threshold; changing this property also updates the slope used in
        the fixed and floating point implementation, and the soft knee
        fitting.
        """
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = value
        self.slope, self.slope_f32 = drcu.rms_compressor_slope_from_ratio(self.ratio)
        self.piecewise_calc()

    def piecewise_calc(self):
        """Calculate the piecewise linear approximation of the soft knee.

        The knee is approximated as a straight line between the knee
        start at (x1, y1) and the knee end at (x2, y2) BUT as the
        envelope is RMS², we actually get a curve.

        x2 is modified to be halfway between the threshold and the end
        of the knee, trying to join closer to the true knee end than
        this can result in overshoot (i.e. going above the hard knee
        curve)

        """
        # W is the knee width, increasing the knee width requires taking
        # an nth root of the envelope, so a width of 10 dB has been used
        # to avoid needing a root
        self.w = 10
        self.offset = 1

        # # Alternative knee values for 15 dB wide knee
        # self.w = 15
        # self.offset = 0.5

        # # Alternative knee values for 20 dB wide knee
        # self.w = 20
        # self.offset = 0.25

        # calculate the start and end values of the soft knee
        x1 = self.threshold * utils.db_pow2gain(-self.w / 2)
        y1 = 1
        x2 = ((self.threshold * utils.db_pow2gain(self.w / 2)) + self.threshold) / 2
        y2 = (self.threshold / (x2)) ** self.slope

        # do a straight line fit between (x1, y1) and (x2, y2), with
        # modification of x by offset if required
        self.knee_start = x1
        self.knee_end = x2
        self.knee_a = (y2 - y1) / (x2**self.offset - (x1**self.offset))
        self.knee_b = (y1) - self.knee_a * (x1**self.offset)

        # convert knee approximation to f32 and int32
        self.knee_start_int = utils.int32(min(round(self.knee_start * 2**self.Q_sig), 2**31 - 1))
        self.knee_end_int = utils.int32(min(round(self.knee_end * 2**self.Q_sig), 2**31 - 1))
        self.knee_a_f32 = float32(self.knee_a)
        self.knee_b_f32 = float32(self.knee_b)
        self.knee_b_int = utils.int32((self.knee_b - 1) * 2**31)

    def compressor_rms_softknee_gain_calc(self, envelope, threshold, slope=None):
        """Calculate the float gain for the current sample.

        Note that as the RMS envelope detector returns x², we need to
        use db_pow. The knee is exponential in the log domain, so must
        be calculated in the log domain.

        Below the start of the knee, the gain is 1. Above the end of the
        knee, the gain is the same as a regular RMS compressor.
        """
        envelope_db = utils.db_pow(envelope)
        if envelope_db < (self.threshold_db - self.w / 2):
            new_gain = 1
        elif envelope_db < (self.threshold_db + self.w / 2):
            # soft knee
            new_gain_db = (-self.slope / (self.w)) * (
                envelope_db - self.threshold_db + self.w / 2
            ) ** 2
            new_gain = utils.db2gain(new_gain_db)
        else:
            # regular RMS compressor
            new_gain = (self.threshold / envelope) ** self.slope
        new_gain = min(1, new_gain)
        return new_gain

    def compressor_rms_softknee_gain_calc_approx(self, envelope, threshold, slope=None):
        """Calculate the float gain for the current sample, using a
        linear approximation for the soft knee. Since the RMS envelope
        is used, and returns RMS², the linear approximation gives a
        quadratic fit, and so is reasonably close to the true soft knee.

        Below the start of the knee, the gain is 1. Above the end of the
        knee, the gain is the same as a regular RMS compressor.
        """
        if envelope < self.knee_start:
            new_gain = 1

        elif envelope < self.knee_end:
            # straight line, but envelope is RMS**2, so actually squared
            # 🤯
            new_gain = self.knee_a * (envelope**self.offset) + self.knee_b
        else:
            # regular RMS compressor
            new_gain = (self.threshold / envelope) ** self.slope

        return new_gain

    def compressor_rms_softknee_gain_calc_xcore(self, envelope_int, threshold_int, slope_f32=None):
        """Calculate the int gain for the current sample, using a
        linear approximation for the soft knee. Since the RMS envelope
        is used, and returns RMS², the linear approximation gives a
        quadratic fit, and so is reasonably close to the true soft knee.

        Below the start of the knee, the gain is 1. Above the end of the
        knee, the gain is the same as a regular RMS compressor.

        """
        if envelope_int < self.knee_start_int:
            new_gain_int = utils.int32(0x7FFFFFFF)
            return new_gain_int

        elif envelope_int < self.knee_end_int:
            # Straight line, but envelope is RMS**2, so actually squared
            # This has to be partly done in float32 as knee_a has a
            # really big range that can't be reliably represented in
            # int32.
            # knee_b varies between 1.0 and 1.055, so is represented as
            # 1+knee_b_int (alternatively we could just keep b as f32).
            # knee_a is always negative, so adding b should give a
            # result < 1.
            env_f32 = float32(envelope_int * 2**-self.Q_sig)
            new_gain_f32 = env_f32 * self.knee_a_f32
            new_gain_int = (new_gain_f32 * float32(2**31)).as_int32()
            new_gain_int = utils.int32(new_gain_int + self.knee_b_int + 2**31)

        else:
            # regular RMS compressor
            new_gain_int = int(threshold_int) << 31
            new_gain_int = utils.int32(new_gain_int // envelope_int)
            new_gain_int = (
                (float32(new_gain_int * 2**-31) ** slope_f32) * float32(2**31)
            ).as_int32()

        return new_gain_int
