# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import warnings

from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils


class mixer(dspg.dsp_block):
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
        Gain in decibels.
    gain : float
        Gain as a linear value.
    gain_int : int
        Gain as an integer value.
    """

    def __init__(
        self, fs: float, num_channels: int, gain_db: float = -6, Q_sig: int = dspg.Q_SIG
    ) -> None:
        super().__init__(fs, num_channels, Q_sig)
        self.num_channels = num_channels
        assert gain_db <= 24, "Maximum mixer gain is +24dB"
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
        self.gain_int = utils.int32(self.gain * 2**self.Q_sig)

    def process(self, sample_list: list[float], channel: int = 0) -> float:  # type: ignore
        """
        Process a single sample. Apply the gain to all the input samples
        then sum them using floating point maths.

        Parameters
        ----------
        sample_list : list
            List of input samples
        channel : int
            Channel index, not used by this module.

        Returns
        -------
        float
            Output sample.

        """
        scaled_samples = np.array(sample_list) * self.gain
        y = np.sum(scaled_samples)

        return y

    def process_xcore(self, sample_list: list[float], channel: int = 0) -> float:  # type: ignore
        """
        Process a single sample. Apply the gain to all the input samples
        then sum them using int32 fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting.

        Parameters
        ----------
        sample_list : list
            List of input samples
        channel : int
            Channel index, not used by this module.

        Returns
        -------
        float
            Output sample.

        """
        y = 0
        for sample in sample_list:
            sample_int = utils.int32(round(sample * 2**self.Q_sig))
            scaled_sample = utils.int32_mult_sat_extract(sample_int, self.gain_int, self.Q_sig)
            y = utils.int40(y + scaled_sample)

        y = utils.int32(y)
        y_flt = float(y) * 2**-self.Q_sig

        return y_flt

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
            output[sample] = self.process(frame_np[:, sample].tolist())

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
            output[sample] = self.process_xcore(frame_np[:, sample].tolist())

        return [output]

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

    def __init__(self, fs: float, num_channels: int) -> None:
        super().__init__(fs, num_channels, gain_db=0)


class subtractor(dspg.dsp_block):
    """
    Subtractor class for subtracting two signals.

    """

    def __init__(self, fs: float, Q_sig: int = dspg.Q_SIG) -> None:
        # always has 2 channels
        super().__init__(fs, 2, Q_sig)

    def process(self, sample_list: list[float], channel: int = 0) -> float:  # type: ignore
        """
        Subtract the second input sample from the first using floating
        point maths.

        Parameters
        ----------
        sample_list : list[float]
            List of input samples.
        channel : int
            Channel index, unused by this module.

        Returns
        -------
        float
            Result of the subtraction.
        """
        y = sample_list[0] - sample_list[1]
        return y

    def process_xcore(self, sample_list: list[float], channel: int = 0) -> float:  # type: ignore
        """
        Subtract the second input sample from the first using int32
        fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting

        Parameters
        ----------
        sample_list : list[float]
            List of input samples.
        channel : int
            Channel index, unused by this module.

        Returns
        -------
        float
            Result of the subtraction.
        """
        sample_int_0 = utils.int32(round(sample_list[0] * 2**self.Q_sig))
        sample_int_1 = utils.int32(round(sample_list[1] * 2**self.Q_sig))

        y = utils.int32(sample_int_0 - sample_int_1)
        y_flt = float(y) * 2**-self.Q_sig

        return y_flt

    def process_frame(self, frame: list[np.ndarray], channel: int = 0) -> list[np.ndarray]:
        """
        Process a frame of samples, using floating point maths.

        When subtracting, the input channels are combined into a single
        output channel. This means the output frame will be a list of
        length 1.

        Parameters
        ----------
        frame : list[np.ndarray]
            List of frames, where each frame is a 1-D numpy array.
        channel : int
            Channel index.

        Returns
        -------
        list[np.ndarray]
            Length 1 list of processed frames.
        """
        assert len(frame) == 2, "Subtractor requires 2 channels"
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process([frame[0][sample], frame[1][sample]])

        return [output]

    def process_frame_xcore(self, frame: list[np.ndarray], channel: int = 0) -> list[np.ndarray]:
        """
        Process a frame of samples, using int32 fixed point maths.

        When subtracting, the input channels are combined into a single
        output channel. This means the output frame will be a list of
        length 1.

        Parameters
        ----------
        frame : list[np.ndarray]
            List of frames, where each frame is a 1-D numpy array.
        channel : int
            Channel index.

        Returns
        -------
        list[np.ndarray]
            Length 1 list of processed frames.
        """
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process_xcore([frame[0][sample], frame[1][sample]])

        return [output]


class fixed_gain(dspg.dsp_block):
    """Multiply every sample by a fixed gain value.

    In the current implementation, the maximum boost is 6dB.

    Parameters
    ----------
    gain_db : float
        The gain in decibels. Maximum fixed gain is +24dB.
    gain : float
        Gain as a linear value.
    gain_int : int
        Gain as an integer value.
    """

    def __init__(self, fs: float, n_chans: int, gain_db: float, Q_sig: int = dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)
        assert gain_db <= 24, "Maximum fixed gain is +24dB"
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
        self.gain_int = utils.int32(self.gain * 2**self.Q_sig)

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
        sample_int = utils.int32(round(sample * 2**self.Q_sig))
        y = utils.int32_mult_sat_extract(sample_int, self.gain_int, self.Q_sig)

        y_flt = float(y) * 2**-self.Q_sig

        return y_flt

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


class volume_control(fixed_gain):
    """
    A volume control class that allows setting the gain in decibels.

    Inherits from the `fixed_gain` class.

    """

    def set_gain(self, gain_db: float) -> None:
        """Set the gain of the volume control.

        Parameters
        ----------
        gain_db : float
            The gain in decibels. Must be less than or equal to 24 dB.

        Raises
        ------
        AssertionError
            If the gain_db parameter is greater than 24 dB.

        """
        assert gain_db <= 24, "Maximum volume control gain is +24dB"
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
        self.gain_int = utils.int32(self.gain * 2**30)


class switch(dspg.dsp_block):
    """A class representing a switch in a signal chain.

    Attributes
    ----------
    switch_position : int
        The current position of the switch.

    """

    def __init__(self, fs, n_chans, Q_sig: int = dspg.Q_SIG) -> None:
        super().__init__(fs, n_chans, Q_sig)
        self.switch_position = 0
        return

    def process(self, sample_list: list[float], channel: int = 0) -> float:  # type: ignore
        """Return the sample at the current switch position.

        This method takes a list of samples and returns the sample at
        the current switch position.

        Parameters
        ----------
        sample_list : list
            A list of samples for each of the switch inputs.
        channel : int
            Not used by this DSP module.

        Returns
        -------
        y : float
            The sample at the current switch position.
        """
        y = sample_list[self.switch_position]
        return y

    def process_xcore(self, sample_list: list[float], channel: int = 0) -> float:  # type: ignore
        """Return the sample at the current switch position.

        As there is no DSP, this just calls self.process.

        Parameters
        ----------
        sample_list : list
            A list of samples for each of the switch inputs.
        channel : int
            Not used by this DSP module.

        Returns
        -------
        y : float
            The sample at the current switch position.
        """
        return self.process(sample_list)

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
