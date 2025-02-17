# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for signal chain components and basic maths."""

import numpy as np
import warnings

from audio_dsp import _deprecated
from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils

# Q format for signal gains
Q_GAIN = 27
# Just a remainder
assert Q_GAIN == 27, "Need to change the assert in the mixer and fixed_gain inits"


def db_to_qgain(db_in):
    """Calculate the linear gain in floating and fixed point from a
    target gain in decibels.
    If the gain is higher than 24 dB, it is saturated to that value.
    """
    db_in = _check_gain(db_in)

    if db_in == -np.inf:
        gain = 0
        gain_int = utils.int32(0)
    else:
        gain = utils.db2gain(db_in)
        gain_int = utils.float_to_int32(gain, Q_GAIN)

    return gain, gain_int


def _check_gain(value):
    if value > 24:
        warnings.warn("Maximum gain is +24 dB, saturating to that value.", UserWarning)
        value = 24
    return value


class _combiners(dspg.dsp_block):
    """_combiners take multiple inputs and combine them to one output,
    so the output frame size is different.
    """

    def process_frame(self, frame: list[np.ndarray]) -> list[np.ndarray]:
        """
        Take a list frames of samples and return the processed frames,
        using floating point maths.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When adding, the input channels are combined into a single
        output channel. This means the output frame will be a list of
        length 1.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.

        Returns
        -------
        list
            Length 1 list of processed frames.
        """
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process_channels(frame_np[:, sample].tolist())[0]

        return [output]

    def process_frame_xcore(self, frame: list[np.ndarray]) -> list[np.ndarray]:
        """
        Take a list frames of samples and return the processed frames,
        using int32 fixed point maths.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When adding, the input channels are combined into a single
        output channel. This means the output frame will be a list of
        length 1.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.

        Returns
        -------
        list
            Length 1 list of processed frames.
        """
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process_channels_xcore(frame_np[:, sample].tolist())[0]

        return [output]


class mixer(_combiners):
    """
    Mixer class for adding signals with attenuation to maintain
    headroom.

    Parameters
    ----------
    gain_db : float
        Gain in decibels (default is -6 dB).

    Attributes
    ----------
    gain_db : float
    gain : float
        Gain as a linear value.
    gain_int : int
        Gain as an integer value.
    """

    def __init__(
        self, fs: float, n_chans: int, gain_db: float = -6, Q_sig: int = dspg.Q_SIG
    ) -> None:
        super().__init__(fs, n_chans, Q_sig)
        self.num_channels = n_chans
        self.gain_db = gain_db

    @property
    def gain_db(self):
        """The mixer gain in decibels."""
        return self._gain_db

    @gain_db.setter
    def gain_db(self, value):
        value = _check_gain(value)
        self._gain_db = value
        self.gain, self.gain_int = db_to_qgain(self._gain_db)

    def process_channels(self, sample_list: list[float]) -> list[float]:
        """
        Process a single sample. Apply the gain to all the input samples
        then sum them using floating point maths.

        Parameters
        ----------
        sample_list : list
            List of input samples

        Returns
        -------
        list[float]
            Output sample.

        """
        scaled_samples = np.array(sample_list) * self.gain
        y = float(np.sum(scaled_samples))
        y = utils.saturate_float(y, self.Q_sig)
        return [y]

    def process_channels_xcore(self, sample_list: list[float]) -> list[float]:
        """
        Process a single sample. Apply the gain to all the input samples
        then sum them using int32 fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting.

        Parameters
        ----------
        sample_list : list
            List of input samples

        Returns
        -------
        list[float]
            Output sample.

        """
        y = int(0)
        for sample in sample_list:
            sample_int = utils.float_to_int32(sample, self.Q_sig)
            acc = 1 << (Q_GAIN - 1)
            acc += sample_int * self.gain_int
            scaled_sample = utils.int32_mult_sat_extract(acc, 1, Q_GAIN)
            y += scaled_sample

        y = utils.int32_mult_sat_extract(y, 2, 1)
        y_flt = utils.int32_to_float(y, self.Q_sig)

        return [y_flt]

    def freq_response(self, nfft: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the mixer, assumed to be a
        flat response scaled by the gain.

        Parameters
        ----------
        nfft : int
            Number of FFT points.

        Returns
        -------
        tuple
            A tuple containing the frequency values and the
            corresponding complex response.

        """
        w = np.fft.rfftfreq(nfft)
        h = np.ones_like(w) * self.gain
        return w, h


class adder(mixer):
    """
    A class representing an adder in a signal chain.

    This class inherits from the `mixer` class and provides an adder
    with no attenuation.

    """

    def __init__(self, fs: float, n_chans: int, Q_sig: int = dspg.Q_SIG) -> None:
        super().__init__(fs, n_chans, gain_db=0, Q_sig=Q_sig)


class subtractor(_combiners):
    """Subtractor class for subtracting two signals."""

    def __init__(self, fs: float, Q_sig: int = dspg.Q_SIG) -> None:
        # always has 2 channels
        super().__init__(fs, 2, Q_sig)

    def process_channels(self, sample_list: list[float]) -> list[float]:
        """
        Subtract the second input sample from the first using floating
        point maths.

        Parameters
        ----------
        sample_list : list[float]
            List of input samples.

        Returns
        -------
        float
            Result of the subtraction.
        """
        y = sample_list[0] - sample_list[1]
        y = utils.saturate_float(y, self.Q_sig)
        return [y]

    def process_channels_xcore(self, sample_list: list[float]) -> list[float]:
        """
        Subtract the second input sample from the first using int32
        fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting

        Parameters
        ----------
        sample_list : list[float]
            List of input samples.

        Returns
        -------
        float
            Result of the subtraction.
        """
        sample_int_0 = utils.float_to_int32(sample_list[0], self.Q_sig)
        sample_int_1 = utils.float_to_int32(sample_list[1], self.Q_sig)

        acc = int(0)
        acc += sample_int_0 * 2
        acc += sample_int_1 * -2
        y = utils.int32_mult_sat_extract(acc, 1, 1)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return [y_flt]


class fixed_gain(dspg.dsp_block):
    """Multiply every sample by a fixed gain value.

    In the current implementation, the maximum boost is +24 dB.

    Parameters
    ----------
    gain_db : float
        The gain in decibels. Maximum fixed gain is +24 dB.

    Attributes
    ----------
    gain_db : float
    gain : float
        Gain as a linear value.
    gain_int : int
        Gain as an integer value.
    """

    def __init__(self, fs: float, n_chans: int, gain_db: float, Q_sig: int = dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)
        self.gain_db = gain_db

    @property
    def gain_db(self):
        """The mixer gain in decibels."""
        return self._gain_db

    @gain_db.setter
    def gain_db(self, value):
        value = _check_gain(value)
        self._gain_db = value
        self.gain, self.gain_int = db_to_qgain(self._gain_db)

    def process(self, sample: float, channel: int = 0) -> float:
        """Multiply the input sample by the gain, using floating point
        maths.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on, not used by this
            module.

        Returns
        -------
        float
            The processed output sample.
        """
        y = sample * self.gain
        y = utils.saturate_float(y, self.Q_sig)

        return y

    def process_xcore(self, sample: float, channel: int = 0) -> float:
        """Multiply the input sample by the gain, using int32 fixed
        point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on, not used by this
            module.

        Returns
        -------
        float
            The processed output sample.
        """
        if isinstance(sample, float):
            sample_int = utils.float_to_int32(sample, self.Q_sig)
        elif isinstance(sample, int):
            sample_int = sample
        else:
            raise TypeError("input must be float or int")

        # for rounding
        acc = 1 << (Q_GAIN - 1)
        acc += sample_int * self.gain_int
        y = utils.int32_mult_sat_extract(acc, 1, Q_GAIN)

        if isinstance(sample, float):
            return utils.int32_to_float(y, self.Q_sig)
        else:
            return y

    def freq_response(self, nfft: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the gain, assumed to be a
        flat response scaled by the gain.

        Parameters
        ----------
        nfft : int
            Number of FFT points.

        Returns
        -------
        tuple
            A tuple containing the frequency values and the
            corresponding complex response.

        """
        w = np.fft.rfftfreq(nfft)
        h = np.ones_like(w) * self.gain
        return w, h


class volume_control(dspg.dsp_block):
    """
    A volume control class that allows setting the gain in decibels.
    When the gain is updated, an exponential slew is applied to reduce
    artifacts.

    The slew is implemented as a shift operation. The slew rate can be
    converted to a time constant using the formula:
    `time_constant = -1/ln(1 - 2^-slew_shift) * (1/fs)`

    A table of the first 10 slew shifts is shown below:

    +------------+--------------------+
    | slew_shift | Time constant (ms) |
    +============+====================+
    |      1     |        0.03        |
    +------------+--------------------+
    |      2     |        0.07        |
    +------------+--------------------+
    |      3     |        0.16        |
    +------------+--------------------+
    |      4     |        0.32        |
    +------------+--------------------+
    |      5     |        0.66        |
    +------------+--------------------+
    |      6     |        1.32        |
    +------------+--------------------+
    |      7     |        2.66        |
    +------------+--------------------+
    |      8     |        5.32        |
    +------------+--------------------+
    |      9     |       10.66        |
    +------------+--------------------+
    |     10     |       21.32        |
    +------------+--------------------+

    Parameters
    ----------
    gain_db : float, optional
        The initial gain in decibels
    slew_shift : int, optional
        The shift value used in the exponential slew.
    mute_state : int, optional
        The mute state of the Volume Control: 0: unmuted, 1: muted.

    Attributes
    ----------
    target_gain_db : float
    target_gain : float
        The target gain as a linear value.
    target_gain_int : int
        The target gain as a fixed-point integer value.
    gain_db : float
        The current gain in decibels.
    gain : float
        The current gain as a linear value.
    gain_int : int
        The current gain as a fixed-point integer value.
    slew_shift : int
        The shift value used in the exponential slew.
    mute_state : int
        The mute state of the Volume Control: 0: unmuted, 1: muted

    Raises
    ------
    ValueError
        If the gain_db parameter is greater than 24 dB.

    """

    def __init__(
        self,
        fs: float,
        n_chans: int,
        gain_db: float = -6,
        slew_shift: int = 7,
        mute_state: int = 0,
        Q_sig: int = dspg.Q_SIG,
    ) -> None:
        super().__init__(fs, n_chans, Q_sig)

        # set the initial target gains
        self.mute_state = False
        self.target_gain_db = gain_db
        if mute_state:
            self.mute()

        # initial applied gain can be equal to target until target changes
        self.gain_db = self.target_gain_db
        self.gain = [self.target_gain] * self.n_chans
        self.gain_int = [self.target_gain_int] * self.n_chans

        self.slew_shift = slew_shift

    @property
    def target_gain_db(self):
        """The target gain in decibels."""
        return self._target_gain_db

    @target_gain_db.setter
    def target_gain_db(self, value):
        value = _check_gain(value)
        if not self.mute_state:
            self._target_gain_db = value
            self.target_gain, self.target_gain_int = db_to_qgain(self._target_gain_db)
        else:
            self.saved_gain_db = value

    def process(self, sample: float, channel: int = 0) -> float:
        """
        Update the current gain, then multiply the input sample by
        it, using floating point maths.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int, optional
            The channel index to process the sample on. Not used by this module.

        Returns
        -------
        float
            The processed output sample.
        """
        # do the exponential slew
        self.gain[channel] += (self.target_gain - self.gain[channel]) * 2**-self.slew_shift

        y = sample * self.gain[channel]
        y = utils.saturate_float(y, self.Q_sig)
        return y

    def process_xcore(self, sample: float, channel: int = 0) -> float:
        """
        Update the current gain, then multiply the input sample by
        it, using int32 fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on, not used by this
            module.

        Returns
        -------
        float
            The processed output sample.
        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        # do the exponential slew
        self.gain_int[channel] += (
            self.target_gain_int - self.gain_int[channel]
        ) >> self.slew_shift

        # print(f"gain {self.gain_int[0]}")

        # for rounding
        acc = 1 << (Q_GAIN - 1)
        acc += sample_int * self.gain_int[channel]
        y = utils.int32_mult_sat_extract(acc, 1, Q_GAIN)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt

    @_deprecated(
        "1.0.0",
        "2.0.0",
        "Replace `volume_control.set_gain(x)` with `volume_control.target_gain_db = x`",
    )
    def set_gain(self, gain_db: float) -> None:
        """
        Set the gain of the volume control.

        Parameters
        ----------
        gain_db : float
            The gain in decibels. Must be less than or equal to 24 dB.

        Raises
        ------
        ValueError
            If the gain_db parameter is greater than 24 dB.

        """
        self.target_gain_db = gain_db

    def mute(self) -> None:
        """Mute the volume control."""
        if not self.mute_state:
            self.saved_gain_db = self.target_gain_db
            self.target_gain_db = -np.inf
            self.mute_state = True

    def unmute(self) -> None:
        """Unmute the volume control."""
        if self.mute_state:
            self.mute_state = False
            self.target_gain_db = self.saved_gain_db


class switch(_combiners):
    """A class representing a switch in a signal chain.

    Attributes
    ----------
    switch_position : int
        The current position of the switch.

    """

    def __init__(self, fs, n_chans, Q_sig: int = dspg.Q_SIG) -> None:
        super().__init__(fs, n_chans, Q_sig)
        self.switch_position = 0
        # don't need separate implementation for float/xcore
        self.process_channels_xcore = self.process_channels
        return

    def process_channels(self, sample_list: list[float]) -> list[float]:
        """Return the sample at the current switch position.

        This method takes a list of samples and returns the sample at
        the current switch position.

        Parameters
        ----------
        sample_list : list
            A list of samples for each of the switch inputs.

        Returns
        -------
        y : list[float]
            The sample at the current switch position.
        """
        y = sample_list[self.switch_position]
        return [y]

    def move_switch(self, position: int) -> None:
        """Move the switch to the specified position. This will cause
        the channel in sample_list[position] to be output.

        Parameters
        ----------
        position : int
            The position to move the switch to.
        """
        if position < 0 or position >= self.n_chans:
            warnings.warn(
                f"Switch position {position} is out of range, keeping old switch position"
            )
        else:
            self.switch_position = position
        return


class switch_stereo(dspg.dsp_block):
    """A class representing a stereo switch in a signal chain.

    The inputs should be grouped in stereo pairs, e.g.
    ``[0_L, 0_R, 1_L, 1_R, ...]``.

    Attributes
    ----------
    switch_position : int
        The current position of the switch.

    """

    def __init__(self, fs, n_chans, Q_sig: int = dspg.Q_SIG) -> None:
        super().__init__(fs, n_chans, Q_sig)
        assert n_chans % 2 == 0
        self.switch_position = 0
        # don't need separate implementation for float/xcore
        self.process_channels_xcore = self.process_channels
        return

    def process_channels(self, sample_list: list[float]) -> list[float]:
        """Return the stereo samples at the current switch position.

        This method takes a list of samples and returns the sample at
        the current switch position.

        Parameters
        ----------
        sample_list : list
            A list of samples for each of the stereo switch inputs.

        Returns
        -------
        y : float
            The stereo samples at the current switch position.
        """
        y = sample_list[(2 * self.switch_position) : (2 * self.switch_position + 2)]
        return y

    def process_frame(self, frame: list[np.ndarray]) -> list[np.ndarray]:
        """
        Take a list frames of samples and return the processed frames,
        using floating point maths.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When switching, the stereo input channels pairs are switched
        between to select the stereo output channel. This means the
        output frame will be a list of length 2.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.
            Stereo pairs should be consecutive.

        Returns
        -------
        list
            Length 2 list of processed frames.
        """
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = [np.zeros(frame_size)] * 2
        for sample in range(frame_size):
            out_samples = self.process_channels(frame_np[:, sample].tolist())
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output

    def process_frame_xcore(self, frame: list[np.ndarray]) -> list[np.ndarray]:
        """
        Take a list frames of samples and return the processed frames,
        using int32 fixed point maths.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When switching, the stereo input channels pairs are switched
        between to select the stereo output channel. This means the
        output frame will be a list of length 2.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.
            Stereo pairs should be consecutive.

        Returns
        -------
        list
            Length 2 list of processed frames.
        """
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = [np.zeros(frame_size)] * 2
        for sample in range(frame_size):
            out_samples = self.process_channels_xcore(frame_np[:, sample].tolist())
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]
        return output

    def move_switch(self, position: int) -> None:
        """Move the switch to the specified position. This will cause
        the channel in sample_list[position] to be output.

        Parameters
        ----------
        position : int
            The position to move the switch to.
        """
        if position < 0 or position >= self.n_chans:
            warnings.warn(
                f"Switch position {position} is out of range, keeping old switch position"
            )
        else:
            self.switch_position = position
        return


class delay(dspg.dsp_block):
    """
    A simple delay line class.

    Note the minimum delay provided by this block is 1 sample. Setting
    the delay to 0 will still yield a 1 sample delay.

    Parameters
    ----------
    max_delay : float
        The maximum delay in specified units.
    starting_delay : float
        The starting delay in specified units.
    units : str, optional
        The units of the delay, can be 'samples', 'ms' or 's'.
        Default is 'samples'.

    Attributes
    ----------
    max_delay : int
        The maximum delay in samples.
    delay : int
        The current delay in samples.
    buffer : np.ndarray
        The delay line buffer.
    buffer_idx : int
        The current index of the buffer.
    """

    def __init__(
        self, fs, n_chans, max_delay: float, starting_delay: float, units: str = "samples"
    ) -> None:
        super().__init__(fs, n_chans)

        self._delay_units = utils.check_time_units(units)
        if max_delay <= 0:
            raise ValueError("Max delay must be greater than zero")

        # max delay cannot be changed, or you'll overflow the buffer
        max_delay = utils.time_to_samples(self.fs, max_delay, self.delay_units)
        self._max_delay = max_delay

        self.delay_time = starting_delay

        self.buffer_idx = 0
        # don't need separate implementation for float/xcore
        self.process_channels_xcore = self.process_channels

        self.reset_state()

    @property
    def delay_time(self):
        """The delay time in delay_units."""
        return self._delay_time

    @delay_time.setter
    def delay_time(self, value):
        self._delay_time = value
        self.delay = utils.time_to_samples(self.fs, value, self.delay_units)

    @property
    def delay_units(self):
        """The units for delay_time. This must be on of {"samples", "ms", "s"}."""
        return self._delay_units

    @delay_units.setter
    def delay_units(self, value):
        self._delay_units = utils.check_time_units(value)

        if self._delay_units == "samples":
            new_coeff = 1
        elif self._delay_units == "ms":
            new_coeff = self.fs / 1000
        elif self._delay_units == "s":
            new_coeff = self.fs

        # keep the same delay in samples, but update delay_time for the new units
        self._delay_time = self.delay * new_coeff

    @property
    def delay(self):
        """The delay in samples. This will saturate to max_delay."""
        return self._delay

    @delay.setter
    def delay(self, value):
        if value <= self._max_delay:
            self._delay = value
        else:
            self._delay = self._max_delay

            warnings.warn(
                "Delay cannot be greater than max delay, setting to max delay", UserWarning
            )

    def reset_state(self) -> None:
        """Reset all the delay line values to zero."""
        self.buffer = np.zeros((self.n_chans, self._max_delay))
        return

    def set_delay(self, delay: float, units: str = "samples") -> None:
        """
        Set the length of the delay line, will saturate at max_delay.

        Parameters
        ----------
        delay : float
            The delay length in specified units.
        units : str, optional
            The units of the delay, can be 'samples', 'ms' or 's'.
            Default is 'samples'.
        """
        # update private units first to avoid recalculating old delay time
        self._delay_units = utils.check_time_units(units)
        self.delay_time = delay
        return

    def process_channels(self, sample_list: list[float]) -> list[float]:
        """
        Put the new sample in the buffer and return the oldest sample.

        Parameters
        ----------
        sample : list
            List of input samples

        Returns
        -------
        float
            List of delayed samples.
        """
        y = self.buffer[:, self.buffer_idx].copy().astype(type(sample_list[0]))
        self.buffer[:, self.buffer_idx] = sample_list
        # not using the modulo because it breaks for when delay = 0
        self.buffer_idx += 1
        if self.buffer_idx >= self.delay:
            self.buffer_idx = 0
        return y.tolist()
