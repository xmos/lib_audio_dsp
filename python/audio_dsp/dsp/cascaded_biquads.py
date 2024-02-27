# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import matplotlib.pyplot as plt

from . import biquad as bq
from . import utils as utils
from audio_dsp.dsp import generic as dspg


class cascaded_biquads_8(dspg.dsp_block):
    """A class representing a cascaded biquad filter with up to 8
    biquads.

    This can be used to either implement a parametric equaliser or a
    higher order filter built out of second order sections.

    8 biquad objects are always created, if there are less than 8
    biquads in the cascade, the remaining biquads are set to bypass
    (b0 = 1).

    Parameters
    ----------
    coeffs_list : list
        List of coefficients for each biquad in the cascade.

    Attributes
    ----------
    biquads : list
        List of biquad objects representing each biquad in the cascade.

    """

    def __init__(self, coeffs_list, fs, n_chans, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)
        self.biquads = [None] * 8
        for n in range(8):
            if n < len(coeffs_list):
                self.biquads[n] = bq.biquad(coeffs_list[n], fs, n_chans)
            else:
                self.biquads[n] = bq.biquad_bypass(fs, n_chans)

    def process(self, sample, channel=0):
        """Process the input sample through the cascaded biquads using
        floating point maths.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on.

        Returns
        -------
        float
            The processed output sample.
        """
        y = sample
        for biquad in self.biquads:
            y = biquad.process(y, channel)

        return y

    def process_frame(self, frame):
        """
        Take a list frames of samples and return the processed frames
        using floating point maths.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        The all the samples are run through each biquad in turn.

        Parameters
        ----------
        frame : list
            List of frames, where each frame is a 1-D numpy array.

        Returns
        -------
        list
            List of processed frames, with the same structure as the
            input frame.
        """
        y = frame
        for biquad in self.biquads:
            y = biquad.process_frame(y)

        return y

    def process_int(self, sample, channel=0):
        """Process the input sample through the cascaded biquads using
        int32 fixed point maths.

        The float input sample is quantized to int32, and returned to
        float before outputting
        """
        y = sample
        for biquad in self.biquads:
            y = biquad.process_int(y, channel)

        return y

    def process_frame_int(self, frame):
        """
        Take a list frames of samples and return the processed frames
        using an int32 fixed point implementation.

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
            List of processed frames, with the same structure as the
            input frame.
        """
        # in the future we could use a more efficient implementation
        y = frame
        for biquad in self.biquads:
            y = biquad.process_frame_int(y)

        return y

    def process_xcore(self, sample, channel=0):
        """Process the input sample through the cascaded biquads using
        int32 fixed point maths, with use of the XS3 VPU

        The float input sample is quantized to int32, and returned to
        float before outputting
        """
        # in the future we could use a more efficient implementation
        y = sample
        for biquad in self.biquads:
            y = biquad.process_xcore(y, channel)

        return y

    def process_frame_xcore(self, frame):
        """
        Take a list frames of samples and return the processed frames
        using int32 fixed point maths, with use of the XS3 VPU

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
            List of processed frames, with the same structure as the
            input frame.
        """
        # in the future we could use a more efficient implementation
        y = frame
        for biquad in self.biquads:
            y = biquad.process_frame_xcore(y)

        return y

    def freq_response(self, nfft=512):
        """
        Calculate the frequency response of the cascaded biquad filters.

        The biquad filter coefficients for each biquad are scaled and
        returned to numerator and denominator coefficients, before being
        passed to `scipy.signal.freqz` to calculate the frequency
        response.

        The stages are then combined by multiplying the complex
        frequency responses.

        Parameters
        ----------
        nfft : int
            The number of points to compute in the frequency response,
            by default 512.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing the frequency vector and the complex
            frequency response.

        """
        f, h_all = self.biquads[0].freq_response(nfft)
        for biquad in self.biquads[1:]:
            _, h = biquad.freq_response(nfft)
            h_all *= h

        return f, h_all

    def reset_state(self):
        """
        Reset the biquad saved states to zero.
        """
        for biquad in self.biquads:
            biquad.reset_state()

        return


class butterworth_lowpass(cascaded_biquads_8):
    """A Butterworth lowpass filter implementation using cascaded
    biquads.

    Parameters
    ----------
    N : int
        The order of the Butterworth filter.
    fc : float
        The cutoff frequency of the filter.
    """

    def __init__(self, fs, n_chans, N, fc):
        coeffs_list = make_butterworth_lowpass(N, fc, fs)
        super().__init__(coeffs_list, fs, n_chans)


class butterworth_highpass(cascaded_biquads_8):
    """A Butterworth highpass filter implementation using cascaded
    biquads.

    Parameters
    ----------
    N : int
        The order of the Butterworth filter.
    fc : float
        The cutoff frequency of the filter.
    """

    def __init__(self, fs, n_chans, N, fc):
        coeffs_list = make_butterworth_highpass(N, fc, fs)
        super().__init__(coeffs_list, fs, n_chans)


class parametric_eq_8band(cascaded_biquads_8):
    """A parametric equalizer with 8 bands.

    This class extends the `cascaded_biquads_8` class to implement a
    parametric equalizer with 8 bands. It applies a series of cascaded
    biquad filters to the audio signal.

    Parameters
    ----------
    filter_spec : list
        A list of tuples specifying the filter parameters for each band.
        Each tuple should contain the filter type as a string (e.g.,
        'lowpass', 'highpass', 'peaking'), followed by the filter
        parameters specific to that type.
    """

    def __init__(self, fs, n_chans, filter_spec):
        coeffs_list = []
        for spec in filter_spec:
            class_name = f"make_biquad_{spec[0]}"
            class_handle = getattr(bq, class_name)
            coeffs_list.append(class_handle(fs, *spec[1:]))

        super().__init__(coeffs_list, fs, n_chans)


def make_butterworth_lowpass(N, fc, fs):
    """
    Generate N/2 sets of biquad coefficients for a Butterworth low-pass
    filter.

    The function implements the algorithm described in Neil Robertson's
    article:
    "Designing Cascaded Biquad Filters Using the Pole-Zero Method"
    `https://www.dsprelated.com/showarticle/1137.php`_

    It uses the bilinear transform to convert the analog filter poles to
    the z-plane.

    See also `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html`_

    Parameters
    ----------
    N : int
        Filter order (must be even).
    fc : float
        -3 dB frequency in Hz.
    fs : float
        Sample frequency in Hz.

    Returns
    -------
    list
        A list of N/2 sets of biquad coefficients, where each set contains the
        coefficients (b0, b1, b2, a0, a1, a2) for a biquad filter.

    Raises
    ------
    AssertionError
        If fc is greater than fs/2 or if N is not even.
    """
    assert fc <= fs / 2, "fc must be less than fs/2"
    assert N % 2 == 0, "N must be even"

    # Find analog filter poles above the real axis for the low-pass
    ks = np.arange(1, N // 2 + 1)
    theta = (2 * ks - 1) * np.pi / (2 * N)
    pa = -np.sin(theta) + 1j * np.cos(theta)
    # reverse sequence of poles – put high Q last to minimise change of clipping
    pa = np.flip(pa)

    # scale poles in frequency
    Fc = fs / np.pi * np.tan(np.pi * fc / fs)
    pa = pa * 2 * np.pi * Fc

    # poles in the z plane by bilinear transform
    p = (1 + pa / (2 * fs)) / (1 - pa / (2 * fs))

    coeffs_list = []
    for k in ks:
        # denominator coefficients
        a0 = 1
        a1 = -2 * np.real(p[k - 1])
        a2 = abs(p[k - 1]) ** 2

        # numerator coefficients
        K = (a0 + a1 + a2) / 4
        b0 = K
        b1 = 2 * K
        b2 = K

        coeffs = (b0, b1, b2, a0, a1, a2)
        coeffs = bq.normalise_biquad(coeffs)
        coeffs_list.append(coeffs)

    return coeffs_list


def make_butterworth_highpass(N, fc, fs):
    """
    Generate N/2 sets of biquad coefficients for a Butterworth high-pass
    filter.

    The function implements the algorithm described in Neil Robertson's
    article:
    "Designing Cascaded Biquad Filters Using the Pole-Zero Method"
    (`https://www.dsprelated.com/showarticle/1137.php`_) and
    "Design IIR Highpass Filters"
    `https://www.dsprelated.com/showarticle/1135.php`_

    It uses the bilinear transform to convert the analog filter poles to
    the z-plane.

    See also `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html`_

    Parameters
    ----------
    N : int
        Filter order (must be even).
    fc : float
        -3 dB frequency in Hz.
    fs : float
        Sample frequency in Hz.

    Returns
    -------
    list
        A list of N/2 sets of biquad coefficients, where each set contains the
        coefficients (b0, b1, b2, a0, a1, a2) for a biquad filter.

    Raises
    ------
    AssertionError
        If fc is greater than fs/2 or if N is not even.
    """

    assert fc <= fs / 2, "fc must be less than fs/2"
    assert N % 2 == 0, "N must be even"

    # Find analog filter poles above the real axis for the low-pass
    ks = np.arange(1, N // 2 + 1)
    theta = (2 * ks - 1) * np.pi / (2 * N)
    pa = -np.sin(theta) + 1j * np.cos(theta)
    # reverse sequence of poles – put high Q last to minimise change of clipping
    pa = np.flip(pa)

    # scale poles in frequency
    Fc = fs / np.pi * np.tan(np.pi * fc / fs)
    # transform to hp poles
    pa = 2 * np.pi * Fc / pa

    # poles in the z plane by bilinear transform
    p = (1 + pa / (2 * fs)) / (1 - pa / (2 * fs))

    coeffs_list = []
    for k in ks:
        # denominator coefficients
        a0 = 1
        a1 = -2 * np.real(p[k - 1])
        a2 = abs(p[k - 1]) ** 2

        # numerator coefficients
        K = (a0 - a1 + a2) / 4
        b0 = K
        b1 = -2 * K
        b2 = K

        coeffs = (b0, b1, b2, a0, a1, a2)
        coeffs = bq.normalise_biquad(coeffs)
        coeffs_list.append(coeffs)

    return coeffs_list


if __name__ == "__main__":
    fs = 48000
    filter_spec = [["lowpass", 8000, 0.707], ["highpass", 200, 1], ["peaking", 1000, 5, 10]]
    peq = parametric_eq_8band(fs, 1, filter_spec)

    w, response = peq.freq_response()

    fig, figs = plt.subplots(2, 1)
    figs[0].semilogx(w / (2 * np.pi) * fs, utils.db(response))
    figs[0].grid()

    figs[1].semilogx(w / (2 * np.pi) * fs, np.angle(response))
    figs[1].grid()

    plt.show()

    fc = 6.7
    fs = 100

    a = make_butterworth_highpass(6, fc, fs)

    bq0 = bq.biquad(bq.normalise_biquad(a[0]), fs, 1)
    bq1 = bq.biquad(bq.normalise_biquad(a[1]), fs, 1)
    bq2 = bq.biquad(bq.normalise_biquad(a[2]), fs, 1)

    w, response0 = bq0.freq_response()
    w, response1 = bq1.freq_response()
    w, response2 = bq2.freq_response()
    response = response0 * response1 * response2

    fig, figs = plt.subplots(2, 1)
    figs[0].plot(w / (2 * np.pi) * fs, utils.db(response0))
    figs[0].plot(w / (2 * np.pi) * fs, utils.db(response1))
    figs[0].plot(w / (2 * np.pi) * fs, utils.db(response2))
    figs[0].plot(w / (2 * np.pi) * fs, utils.db(response))
    figs[0].grid()

    figs[1].plot(w / (2 * np.pi) * fs, np.angle(response))
    figs[1].grid()

    plt.show()

    pass
