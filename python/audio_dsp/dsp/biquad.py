# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The biquad DSP block."""

import warnings
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import scipy.signal as spsig
import matplotlib.pyplot as plt
from docstring_inheritance import inherit_numpy_docstring

from audio_dsp.dsp import utils as utils
from audio_dsp.dsp import generic as dspg

BOOST_BSHIFT = 2  # limit boosts to 12 dB gain


class biquad(dspg.dsp_block):
    """
    A second order biquadratic filter instance.

    This implements a direct form 1 biquad filter, using the
    coefficients provided at initialisation:
    `a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`

    For efficiency the biquad coefficients are normalised by a0 and the
    output `a` coefficients multiplied by -1.

    When the coefficients are updated, the biquad states are reset. This
    helps avoid large errors, but can make this implementation unsuitable
    for real time control. For real time control, :py:class:`biquad_slew` may be a
    better choice.

    Parameters
    ----------
    coeffs : list[float]
        List of normalised biquad coefficients in the form in the form
        `[b0, b1, b2, -a1, -a2]/a0`
    b_shift : int
        The number of right shift bits applied to the b coefficients.
        The default coefficient scaling allows for a maximum coefficient
        value of 2, but high gain shelf and peaking filters can have
        coefficients above this value. Shifting the b coefficients down
        allows coefficients greater than 2, with the cost of b_shift
        bits of precision.

    Attributes
    ----------
    coeffs : list[float]
        List of normalised float biquad coefficients in the form in the
        form `[b0, b1, b2, -a1, -a2]/a0`, rounded to int32 precision.
    int_coeffs : list[int]
        List of normalised int biquad coefficients in the form in the
        form `[b0, b1, b2, -a1, -a2]/a0`, scaled and rounded to int32.

    """

    def __init__(
        self,
        coeffs: list[float],
        fs: int,
        n_chans: int = 1,
        b_shift: int = 0,
        Q_sig: int = dspg.Q_SIG,
    ):
        super().__init__(fs, n_chans, Q_sig)

        self.b_shift = b_shift

        # coeffs should be in the form [b0 b1 b2 -a1 -a2], and
        # normalized by a0
        self.coeffs, self.int_coeffs = _round_and_check(coeffs, self.b_shift)

        self._check_gain()

        # state variables. Note these are shared between float and int
        # implementations, and so the list may become a list of ints!
        self._x1 = [0.0] * n_chans
        self._x2 = [0.0] * n_chans
        self._y1 = [0.0] * n_chans
        self._y2 = [0.0] * n_chans

    def update_coeffs(self, new_coeffs: list[float]):
        """Update the saved coefficients to the input values.

        Parameters
        ----------
        new_coeffs : list[float]
            The new coefficients to be updated.
        """
        self.coeffs, self.int_coeffs = _round_and_check(new_coeffs, self.b_shift)

        # reset states to avoid clicks
        self._x1 = [0.0] * self.n_chans
        self._x2 = [0.0] * self.n_chans
        self._y1 = [0.0] * self.n_chans
        self._y2 = [0.0] * self.n_chans

    def process(self, sample: float, channel: int = 0) -> float:
        """
        Filter a single sample using direct form 1 biquad using floating
        point maths.

        """
        y = (
            self.coeffs[0] * sample
            + self.coeffs[1] * self._x1[channel]
            + self.coeffs[2] * self._x2[channel]
            + self.coeffs[3] * self._y1[channel]
            + self.coeffs[4] * self._y2[channel]
        )

        y = utils.saturate_float(y, self.Q_sig)

        self._x2[channel] = self._x1[channel]
        self._x1[channel] = sample
        self._y2[channel] = self._y1[channel]
        self._y1[channel] = y

        y = y * (1 << self.b_shift)
        y = utils.saturate_float(y, self.Q_sig)

        return y

    def process_int(self, sample: float, channel: int = 0) -> float:
        """
        Filter a single sample using direct form 1 biquad using int32
        fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting

        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        # process a single sample using direct form 1
        y = utils.int64(
            sample_int * self.int_coeffs[0]
            + self._x1[channel] * self.int_coeffs[1]
            + self._x2[channel] * self.int_coeffs[2]
            + self._y1[channel] * self.int_coeffs[3]
            + self._y2[channel] * self.int_coeffs[4]
        )

        # the b_shift can be combined with the >> 30, which reduces
        # quantization noise, but this results  in saturation at an
        # earlier point, and so is not used here for consistency
        y = utils.int64(y + (1 << 29))

        y = utils.int32_mult_sat_extract(y, 1, 30)
        # save states
        self._x2[channel] = utils.int32(self._x1[channel])
        self._x1[channel] = utils.int32(sample_int)
        self._y2[channel] = utils.int32(self._y1[channel])
        self._y1[channel] = utils.int32(y)

        # compensate for coefficients
        y = utils.int64(y << self.b_shift)
        y = utils.saturate_int32(y)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt

    def process_xcore(self, sample: float, channel: int = 0) -> float:
        """
        Filter a single sample using direct form 1 biquad using int32
        fixed point maths, with use of the XS3 VPU.

        The float input sample is quantized to int32, and returned to
        float before outputting.

        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        # process a single sample using direct form 1. In the VPU the
        # ``>> 30`` comes before accumulation
        y = utils.vlmaccr(
            [
                sample_int,
                self._x1[channel],
                self._x2[channel],
                self._y1[channel],
                self._y2[channel],
            ],
            self.int_coeffs,
        )

        y = utils.saturate_int32(y)

        # save states
        self._x2[channel] = utils.int32(self._x1[channel])
        self._x1[channel] = utils.int32(sample_int)
        self._y2[channel] = utils.int32(self._y1[channel])
        self._y1[channel] = utils.int32(y)

        # compensate for coefficients
        y = utils.int64(y << self.b_shift)
        y = utils.saturate_int32(y)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt

    def process_frame_int(self, frame: list[np.ndarray]) -> list[np.ndarray]:
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact int implementation.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.

        Returns
        -------
        list
            List of processed frames, with the same structure as the input frame.
        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_int(this_chan[sample], channel=chan)

        return output

    def freq_response(
        self, nfft: int = 1024
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calculate the frequency response of the biquad filter.

        The biquad filter coefficients are scaled and returned to
        numerator and denominator coefficients, before being passed to
        `scipy.signal.freqz` to calculate the frequency response.

        Parameters
        ----------
        nfft : int
            The number of points to compute in the frequency response,
            by default 1024.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing the frequency vector and the complex
            frequency response.

        """
        b = [self.coeffs[0], self.coeffs[1], self.coeffs[2]]
        b = _apply_biquad_bshift(b, -self.b_shift)
        a = [1, -self.coeffs[3], -self.coeffs[4]]
        f, h = spsig.freqz(b, a, worN=nfft, fs=self.fs)  # type: ignore  it thinks a is supposed to be an int?

        return f, h

    def _check_gain(self):
        """
        Check if the gain of the biquad filter is greater than the
        available headroom. If so, display a warning.

        """
        _, h = self.freq_response()
        max_gain = np.max(utils.db(h))
        if max_gain > dspg.HEADROOM_DB:
            warnings.warn(
                "biquad gain (%.1f dB) is > headroom" % (max_gain)
                + " (%.0f dB), overflow may occur" % dspg.HEADROOM_DB
                + " unless signal level has previously been reduced"
            )

    def reset_state(self):
        """Reset the biquad saved states to zero."""
        for chan in range(self.n_chans):
            self._x1[chan] = 0
            self._x2[chan] = 0
            self._y1[chan] = 0
            self._y2[chan] = 0


class biquad_slew(biquad):
    """
    A second order biquadratic filter instance that slews between
    coefficient updates.

    This implements a direct form 1 biquad filter, using the
    coefficients provided at initialisation:
    `a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`

    For efficiency the biquad coefficients are normalised by a0 and the
    output `a` coefficients multiplied by -1.

    When the target coefficients are updated, the applied coefficients
    are slewed towards the new target values. This makes this implementation
    suitable for real time control. A table of the first 10 slew shifts is shown below:

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
    slew_shift : int
        The shift value used in the exponential slew.

    Attributes
    ----------
    coeffs : list[list[float]]
        List of normalised float biquad coefficients in the form in the
        form `[b0, b1, b2, -a1, -a2]/a0`, rounded to int32 precision.
    int_coeffs : list[list[int]]
        List of normalised int biquad coefficients in the form in the
        form `[b0, b1, b2, -a1, -a2]/a0`, scaled and rounded to int32.
    target_coeffs : list[float]
        List of normalised float target biquad coefficients in the form in the
        form `[b0, b1, b2, -a1, -a2]/a0`, rounded to int32 precision. The coeffs
        are slewed towards these values.
    target_coeffs_int : list[int]
        List of normalised int target biquad coefficients in the form in the
        form `[b0, b1, b2, -a1, -a2]/a0`, scaled and rounded to int32. The int_coeffs
        are slewed towards these values.

    """

    def __init__(
        self,
        coeffs: list[float],
        fs: int,
        n_chans: int = 1,
        b_shift: int = 0,
        slew_shift: int = 2,
        Q_sig: int = dspg.Q_SIG,
    ):
        super().__init__(
            coeffs,
            fs,
            n_chans,
            b_shift,
            Q_sig,
        )
        self.target_coeffs = deepcopy(self.coeffs)
        self.target_coeffs_int = deepcopy(self.int_coeffs)
        self.coeffs = [deepcopy(self.coeffs) for chan in range(n_chans)]
        self.int_coeffs = [deepcopy(self.int_coeffs) for chan in range(n_chans)]
        self.slew_shift = slew_shift

    def update_coeffs(self, new_coeffs: list[float]):
        """Update the saved coefficients to the input values.

        Parameters
        ----------
        new_coeffs : list[float]
            The new coefficients to be updated.
        """
        self.target_coeffs, self.target_coeffs_int = _round_and_check(new_coeffs, self.b_shift)

    def process(self, sample: float, channel: int = 0) -> float:
        """
        Filter a single sample using direct form 1 biquad using floating
        point maths. This will slew the coeffs towards the target coefficients.

        """

        for n in range(5):
            self.coeffs[channel][n] += (
                self.target_coeffs[n] - self.coeffs[channel][n]  # pyright: ignore : overloads base class coeffs
            ) * 2**-self.slew_shift

        y = (
            self.coeffs[channel][0] * sample
            + self.coeffs[channel][1] * self._x1[channel]
            + self.coeffs[channel][2] * self._x2[channel]
            + self.coeffs[channel][3] * self._y1[channel]
            + self.coeffs[channel][4] * self._y2[channel]
        )

        y = utils.saturate_float(y, self.Q_sig)

        self._x2[channel] = self._x1[channel]
        self._x1[channel] = sample
        self._y2[channel] = self._y1[channel]
        self._y1[channel] = y

        y = y * (1 << self.b_shift)
        y = utils.saturate_float(y, self.Q_sig)

        return y

    def process_int(self, sample: float, channel: int = 0) -> float:
        """
        Filter a single sample using direct form 1 biquad using int32
        fixed point maths.
        """
        raise NotImplementedError

    def process_xcore(self, sample: float, channel: int = 0) -> float:
        """
        Filter a single sample using direct form 1 biquad using int32
        fixed point maths, with use of the XS3 VPU.

        The float input sample is quantized to int32, and returned to
        float before outputting.

        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        for n in range(5):
            self.int_coeffs[channel][n] += (
                utils.saturate_int32_vpu(self.target_coeffs_int[n] - self.int_coeffs[channel][n])  # pyright: ignore : overloads base class int_coeffs
                >> self.slew_shift
            )

        # process a single sample using direct form 1. In the VPU the
        # ``>> 30`` comes before accumulation
        y = utils.vlmaccr(
            [
                sample_int,
                self._x1[channel],
                self._x2[channel],
                self._y1[channel],
                self._y2[channel],
            ],
            self.int_coeffs[channel],
        )

        y = utils.saturate_int32(y)

        # save states
        self._x2[channel] = utils.int32(self._x1[channel])
        self._x1[channel] = utils.int32(sample_int)
        self._y2[channel] = utils.int32(self._y1[channel])
        self._y1[channel] = utils.int32(y)

        # compensate for coefficients
        y = utils.int64(y << self.b_shift)
        y = utils.saturate_int32(y)

        y_flt = utils.int32_to_float(y, self.Q_sig)

        return y_flt


def biquad_bypass(fs: int, n_chans: int, Q_sig=dspg.Q_SIG) -> biquad:
    """Return a biquad object with `b0 = 1`, i.e. output=input."""
    coeffs = make_biquad_bypass(fs)
    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_gain(fs: int, n_chans: int, gain_db: float, Q_sig=dspg.Q_SIG) -> biquad:
    """Return a biquad object with a fixed linear gain."""
    coeffs = make_biquad_gain(fs, gain_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT, Q_sig=Q_sig)


def biquad_lowpass(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with low pass filter coefficients."""
    coeffs = make_biquad_lowpass(fs, filter_freq, q_factor)
    coeffs = _apply_biquad_gain(coeffs, gain)
    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_highpass(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with high pass filter coefficients."""
    coeffs = make_biquad_highpass(fs, filter_freq, q_factor)
    coeffs = _apply_biquad_gain(coeffs, gain)
    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_bandpass(
    fs: int, n_chans: int, filter_freq: float, bw: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with bandpass filter coefficients."""
    # bw is bandwidth in octaves
    coeffs = make_biquad_bandpass(fs, filter_freq, bw)

    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_bandstop(
    fs: int, n_chans: int, filter_freq: float, bw: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with bandstop filter coefficients."""
    # bw is bandwidth in octaves
    coeffs = make_biquad_bandstop(fs, filter_freq, bw)
    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_notch(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with notch filter coefficients."""
    coeffs = make_biquad_notch(fs, filter_freq, q_factor)
    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_allpass(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with allpass filter coefficients."""
    coeffs = make_biquad_allpass(fs, filter_freq, q_factor)
    return biquad(coeffs, fs, n_chans=n_chans, Q_sig=Q_sig)


def biquad_peaking(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, boost_db: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with peaking filter coefficients."""
    coeffs = make_biquad_peaking(fs, filter_freq, q_factor, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT, Q_sig=Q_sig)


def biquad_constant_q(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, boost_db: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with constant Q peaking filter coefficients."""
    coeffs = make_biquad_constant_q(fs, filter_freq, q_factor, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT, Q_sig=Q_sig)


def biquad_lowshelf(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, boost_db: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with low shelf filter coefficients."""
    # q is similar to standard low pass, i.e. > 0.707 will yield peakiness
    # the level change at f will be boost_db/2
    coeffs = make_biquad_lowshelf(fs, filter_freq, q_factor, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT, Q_sig=Q_sig)


def biquad_highshelf(
    fs: int, n_chans: int, filter_freq: float, q_factor: float, boost_db: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with high shelf filter coefficients."""
    # q is similar to standard high pass, i.e. > 0.707 will yield peakiness
    # the level change at f will be boost_db/2
    coeffs = make_biquad_highshelf(fs, filter_freq, q_factor, boost_db)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=BOOST_BSHIFT, Q_sig=Q_sig)


def biquad_linkwitz(
    fs: int, n_chans: int, f0: float, q0: float, fp: float, qp: float, Q_sig=dspg.Q_SIG
) -> biquad:
    """Return a biquad object with Linkwitz transform filter coefficients."""
    # used for changing one low frequency roll off slope for another,
    # e.g. in a loudspeaker
    coeffs = make_biquad_linkwitz(fs, f0, q0, fp, qp)
    return biquad(coeffs, fs, n_chans=n_chans, b_shift=0, Q_sig=Q_sig)


def _round_to_q30(coeffs: list[float]) -> tuple[list[float], list[int]]:
    """
    Round a list of filter coefficients to Q1.30 format and int32
    precision. The coefficients should already have any b_shift applied.

    Returns the rounded coefficients in float and int formats

    """
    rounded_coeffs = [0.0] * len(coeffs)
    int_coeffs = [0] * len(coeffs)

    Q = 30
    for n in range(len(coeffs)):
        # scale to Q1.30 ints, note this is intentionally not multiplied
        # (2**Q -1) to keep 1.0 as 1.0
        rounded_coeffs[n] = round(coeffs[n] * (1 << Q))
        # check for overflow
        if not (-(1 << 31)) <= rounded_coeffs[n] <= ((1 << 31) - 1):
            raise ValueError(
                "Filter coefficient will overflow (%.4f, %d), reduce gain" % (coeffs[n], n)
            )

        int_coeffs[n] = utils.int32(rounded_coeffs[n])
        # rescale to floats
        rounded_coeffs[n] = rounded_coeffs[n] / (1 << Q)

    return rounded_coeffs, int_coeffs


def _apply_biquad_gain(coeffs: list[float], gain_db: float) -> list[float]:
    """Apply linear gain to the b coefficients."""
    gain = 10 ** (gain_db / 20)
    coeffs[0] = coeffs[0] * gain
    coeffs[1] = coeffs[1] * gain
    coeffs[2] = coeffs[2] * gain

    return coeffs


def _apply_biquad_bshift(coeffs: list[float], b_shift: int) -> list[float]:
    """
    Apply linear bitshift to the b coefficients.

    This can be used for high gain shelf and peaking filters, where the
    filter coefficients are greater than 2, and so cannot be represented
    in Q1.30 format.

    """
    gain = 2**-b_shift
    coeffs[0] = coeffs[0] * gain
    coeffs[1] = coeffs[1] * gain
    coeffs[2] = coeffs[2] * gain

    return coeffs


def _normalise_biquad(coeffs: list[float]) -> list[float]:
    """
    Normalise biquad coefficients by dividing by a0 and making a1 and a2
    negative.

    Expected input format: [b0, b1, b2, a0, a1, a2]
    Expected output format: [b0, b1, b2, -a1, -a2]/a0

    """
    if len(coeffs) != 6:
        raise ValueError("expected list of 6 biquad coefficients")
    # divide by a0, make a1 and a2 negative
    coeffs = [
        coeffs[0] / coeffs[3],
        coeffs[1] / coeffs[3],
        coeffs[2] / coeffs[3],
        -coeffs[4] / coeffs[3],
        -coeffs[5] / coeffs[3],
    ]

    return coeffs


def _round_and_check(coeffs: list[float], b_shift: int = 0) -> tuple[list[float], list[int]]:
    """
    Apply any b_shift to biquad coefficients, then round to int32
    precision, and check the poles are inside the unit circle.

    """
    # round to int32 precision
    if len(coeffs) != 5:
        raise ValueError("coeffs should be in the form [b0 b1 b2 -a1 -a2]")
    coeffs = _apply_biquad_bshift(coeffs, b_shift)
    coeffs, int_coeffs = _round_to_q30(coeffs)

    # check filter is stable
    poles = np.roots([1, -coeffs[3], -coeffs[4]])
    if np.any(np.abs(poles) >= 1):
        raise ValueError("Poles lie outside the unit circle, the filter is unstable")

    return coeffs, int_coeffs


def _check_filter_freq(filter_freq, fs):
    if filter_freq > fs / 2:
        warnings.warn("filter_freq must be less than fs/2, saturating to fs/2", UserWarning)
        filter_freq = fs / 2

    return filter_freq


def _check_max_gain(gain, max_gain):
    if gain > max_gain:
        warnings.warn(
            f"gain_db must be less than {max_gain:.2f}, saturating to {max_gain:.2f}", UserWarning
        )
        gain = max_gain

    return gain


def make_biquad_bypass(fs: int) -> list[float]:
    """
    Create a bypass biquad filter. Only the b0 coefficient is set.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.
    """
    coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float).tolist()
    return coeffs


def make_biquad_mute(fs: int) -> list[float]:
    """
    Create a biquad filter coefficients list that represents a mute
    filter. All the coefficients are 0.

    Parameters
    ----------
    fs : int
        The sampling frequency in Hz.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2].
    """
    coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    return coeffs


def make_biquad_gain(fs: int, gain_db: float) -> list[float]:
    """
    Calculate the coefficients for a biquad filter with a specified
    linear gain.

    Parameters
    ----------
    fs : int
        The sampling frequency in Hz.
    gain_db : float
        The desired gain in decibels.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.
    """
    coeffs = make_biquad_bypass(fs)
    coeffs = _apply_biquad_gain(coeffs, gain_db)

    return coeffs


def make_biquad_lowpass(fs: int, filter_freq: float, q_factor: float) -> list[float]:
    """Create coefficients for a lowpass biquad filter.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.
    filter_freq : float
        The cutoff frequency of the filter.
    q_factor : float
        The Q factor of the filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than fs/2.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2 * q_factor)

    b0 = (+1.0 - np.cos(w0)) / 2.0
    b1 = +1.0 - np.cos(w0)
    b2 = (+1.0 - np.cos(w0)) / 2.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_highpass(fs: int, filter_freq: float, q_factor: float) -> list[float]:
    """Create coefficients for a highpass biquad filter.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.
    filter_freq : float
        The cutoff frequency of the highpass filter.
    q_factor : float
        The Q factor of the highpass filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than fs/2.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2 * q_factor)

    b0 = (1.0 + np.cos(w0)) / 2.0
    b1 = -(1.0 + np.cos(w0))
    b2 = (1.0 + np.cos(w0)) / 2.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


# Constant 0 dB peak gain
def make_biquad_bandpass(fs: int, filter_freq: float, BW) -> list[float]:
    """Create coefficients for a biquad bandpass filter.

    Parameters
    ----------
    fs : int
        The sampling frequency.
    filter_freq : float
        The center frequency of the bandpass filter.
    BW : float
        The bandwidth of the bandpass filter in octaves, measured
        between -3 dB points.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If filter_freq is greater than fs/2.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * BW * w0 / np.sin(w0))

    b0 = alpha
    b1 = +0.0
    b2 = -alpha
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


# Constant 0 dB peak gain
def make_biquad_bandstop(fs: int, filter_freq: float, BW: float) -> list[float]:
    """Create coefficients for a biquad bandstop filter.

    Parameters
    ----------
    fs : int
        The sampling frequency.
    filter_freq : float
        The center frequency of the bandstop filter.
    BW : float
        The bandwidth of the bandstop filter in octaves, measured
        between -3 dB points

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * BW * w0 / np.sin(w0))

    b0 = +1.0
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_notch(fs: int, filter_freq: float, q_factor: float) -> list[float]:
    """Create a biquad notch filter.

    Parameters
    ----------
    fs : int
        The sampling frequency.
    filter_freq : float
        The center frequency of the notch filter.
    q_factor : float
        The Q factor of the notch filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2.0 * q_factor)

    b0 = +1.0
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_allpass(fs: int, filter_freq: float, q_factor: float) -> list[float]:
    """
    Create coefficients for a biquad allpass filter.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.
    filter_freq : float
        The center frequency of the allpass filter.
    q_factor : float
        The Q factor of the allpass filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2.0 * q_factor)

    b0 = +1.0 - alpha
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0 + alpha
    a0 = +1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_peaking(
    fs: int, filter_freq: float, q_factor: float, boost_db: float
) -> list[float]:
    """Create coefficients for a biquad peaking filter.

    Parameters
    ----------
    fs : int
        The sampling frequency in Hz.
    filter_freq : float
        The center frequency of the peaking filter in Hz.
    q_factor : float
        The Q factor of the peaking filter.
    boost_db : float
        The boost in decibels applied by the filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.
    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    max_gain = (BOOST_BSHIFT + 1) * (20 * np.log10(2))
    boost_db = _check_max_gain(boost_db, max_gain)

    A = np.sqrt(10 ** (boost_db / 20))
    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2.0 * q_factor)

    b0 = +1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = +1.0 - alpha * A
    a0 = +1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = +1.0 - alpha / A

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_constant_q(
    fs: int, filter_freq: float, q_factor: float, boost_db: float
) -> list[float]:
    """Create coefficients for a biquad peaking filter with constant Q.

    Constant Q means that the bandwidth of the filter remains constant
    as the gain varies. It is commonly used for graphic equalisers.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.
    filter_freq : float
        The center frequency of the filter in Hz.
    q_factor : float
        The Q factor of the filter.
    boost_db : float
        The boost in decibels applied to the filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.

    References
    ----------
    - Zoelzer, U. (2011). DAFX: Digital Audio Effects. John Wiley & Sons, Table 2.4
    - https://www.musicdsp.org/en/latest/Filters/37-zoelzer-biquad-filters.html

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    max_gain = (BOOST_BSHIFT + 1) * (20 * np.log10(2))
    boost_db = _check_max_gain(boost_db, max_gain)

    V = 10 ** (boost_db / 20)
    w0 = 2.0 * np.pi * filter_freq / fs
    K = np.tan(w0 / 2)

    if boost_db > 0:
        b0 = 1 + V * K / q_factor + K**2
        b1 = 2 * (K**2 - 1)
        b2 = 1 - V * K / q_factor + K**2
        a0 = 1 + K / q_factor + K**2
        a1 = 2 * (K**2 - 1)
        a2 = 1 - K / q_factor + K**2
    else:
        V = 1 / V
        b0 = 1 + (K / q_factor) + K**2
        b1 = 2 * (K**2 - 1)
        b2 = 1 - (K / q_factor) + K**2
        a0 = 1 + (V * K / q_factor) + K**2
        a1 = 2 * (K**2 - 1)
        a2 = 1 - (V * K / q_factor) + K**2

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_lowshelf(
    fs: int, filter_freq: float, q_factor: float, gain_db: float
) -> list[float]:
    """Create coefficients for a lowshelf biquad filter.

    The Q factor is defined in a similar way to standard low pass, i.e.
    > 0.707 will yield peakiness (where the shelf response does not
    monotonically change). The level change at f will be boost_db/2.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.
    filter_freq : float
        The cutoff frequency of the filter.
    q_factor : float
        The Q factor of the filter.
    gain_db : float
        The gain in decibels of the filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    max_gain = (BOOST_BSHIFT) * (20 * np.log10(2))
    gain_db = _check_max_gain(gain_db, max_gain)

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2 * q_factor)

    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_highshelf(
    fs: int, filter_freq: float, q_factor: float, gain_db: float
) -> list[float]:
    """Create coefficients for a highshelf biquad filter.

    The Q factor is defined in a similar way to standard high pass, i.e.
    > 0.707 will yield peakiness. The level change at f will be
    boost_db/2.

    Parameters
    ----------
    fs : int
        The sample rate of the audio signal.
    filter_freq : float
        The cutoff frequency of the filter.
    q_factor : float
        The Q factor of the filter.
    gain_db : float
        The gain in decibels of the filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If the filter frequency is greater than half of the sample rate.

    """
    filter_freq = _check_filter_freq(filter_freq, fs)

    max_gain = (BOOST_BSHIFT) * (20 * np.log10(2))
    gain_db = _check_max_gain(gain_db, max_gain)

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * filter_freq / fs
    alpha = np.sin(w0) / (2 * q_factor)

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


def make_biquad_linkwitz(fs: int, f0: float, q0: float, fp: float, qp: float) -> list[float]:
    """
    Create coefficients for a Linkwitz Transform biquad filter.

    The Linkwitz Transform is commonly used to change the low frequency
    roll off slope of a loudspeaker. When applied to a loudspeaker, it
    will change the cutoff frequency from f0 to fp, and the quality
    factor from q0 to qp.

    Parameters
    ----------
    fs : int
        The sampling frequency of the audio signal.
    f0 : float
        The original cutoff frequency of the filter.
    q0 : float
        The original quality factor of the filter at f0.
    fp : float
        The target cutoff frequency for the filter.
    qp : float
        The target quality factor for the filter.

    Returns
    -------
    list[float]
        The coefficients of the biquad filter in the order
        [b0, b1, b2, -a1, -a2]. The coefficients are normalised by a0
        such that ``a0 = 1``.

    Raises
    ------
    ValueError
        If either f0 or fp is greater than fs/2.

    References
    ----------
    - Linkwitz Transform: https://www.linkwitzlab.com/filters.htm#9
    - Linkwitz Transform in MiniDSP: https://www.minidsp.com/applications/advanced-tools/linkwitz-transform

    """
    f0 = _check_filter_freq(f0, fs)
    fp = _check_filter_freq(fp, fs)

    fc = (f0 + fp) / 2

    # these are translated from the MiniDSP spreadsheet
    d0i = (2 * np.pi * f0) ** 2
    d1i = (2 * np.pi * f0) / q0

    c0i = (2 * np.pi * fp) ** 2
    c1i = (2 * np.pi * fp) / qp

    gn = (2 * np.pi * fc) / (np.tan(np.pi * fc / fs))
    cci = c0i + gn * c1i + (gn**2)

    a0 = cci
    a1 = 2 * (c0i - (gn**2))
    a2 = c0i - gn * c1i + (gn**2)

    b0 = d0i + gn * d1i + (gn**2)
    b1 = 2 * (d0i - (gn**2))
    b2 = d0i - gn * d1i + (gn**2)

    coeffs = [b0, b1, b2, a0, a1, a2]
    coeffs = _normalise_biquad(coeffs)

    return coeffs


# inherit docstrings from make_biquad functions
inherit_numpy_docstring(make_biquad_bypass.__doc__, biquad_bypass)
inherit_numpy_docstring(make_biquad_gain.__doc__, biquad_gain)
inherit_numpy_docstring(make_biquad_lowpass.__doc__, biquad_lowpass)
inherit_numpy_docstring(make_biquad_highpass.__doc__, biquad_highpass)
inherit_numpy_docstring(make_biquad_bandpass.__doc__, biquad_bandpass)
inherit_numpy_docstring(make_biquad_bandstop.__doc__, biquad_bandstop)
inherit_numpy_docstring(make_biquad_notch.__doc__, biquad_notch)
inherit_numpy_docstring(make_biquad_allpass.__doc__, biquad_allpass)
inherit_numpy_docstring(make_biquad_constant_q.__doc__, biquad_constant_q)
inherit_numpy_docstring(make_biquad_lowshelf.__doc__, biquad_lowshelf)
inherit_numpy_docstring(make_biquad_highshelf.__doc__, biquad_highshelf)
inherit_numpy_docstring(make_biquad_linkwitz.__doc__, biquad_linkwitz)

if __name__ == "__main__":
    fs = 48000

    biquad_1 = biquad(make_biquad_notch(fs, 20, 1), 0, Q_sig=30)
    biquad_2 = biquad(make_biquad_notch(fs, 20, 1), 3, Q_sig=30)
    biquad_3 = biquad(make_biquad_notch(fs, 20, 1), 0, Q_sig=27)
    biquad_4 = biquad(make_biquad_notch(fs, 20, 1), 3, Q_sig=27)
    biquad_5 = biquad(make_biquad_notch(fs, 20, 1), 3, Q_sig=27)

    t = np.arange(fs * 4) / fs
    # signal = spsig.chirp(t, 20, 1, 20000, 'log', phi=-90)
    signal = np.sin(2 * np.pi * 997 * t, dtype=np.float64)

    output_1 = np.zeros(len(signal))
    output_2 = np.zeros(len(signal))
    output_3 = np.zeros(len(signal))
    output_4 = np.zeros(len(signal))
    output_5 = np.zeros(len(signal))

    for n in range(len(signal)):
        output_1[n] = biquad_1.process_xcore(signal[n])
        output_2[n] = biquad_2.process_xcore(signal[n])
        output_3[n] = biquad_3.process_xcore(signal[n])
        output_4[n] = biquad_4.process_xcore(signal[n])
        output_5[n] = biquad_5.process(signal[n])

    # plt.plot(signal)
    # plt.plot(output_1)
    # plt.plot(output_2)

    plt.psd(output_1, 1024 * 16, fs, window=spsig.windows.blackmanharris(1024 * 16))
    plt.psd(output_2, 1024 * 16, fs, window=spsig.windows.blackmanharris(1024 * 16))
    plt.psd(output_3, 1024 * 16, fs, window=spsig.windows.blackmanharris(1024 * 16))
    plt.psd(output_4, 1024 * 16, fs, window=spsig.windows.blackmanharris(1024 * 16))
    plt.psd(output_5, 1024 * 16, fs, window=spsig.windows.blackmanharris(1024 * 16))

    ax = plt.gca()
    ax.set_xscale("log")
    plt.legend(["Q30", "Q30, b_shift 3", "Q27", "Q27, b_shift 3", "double"])
    plt.show()

    pass
    exit()
