# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The biquad stage."""

from ..design.stage import Stage, find_config
import audio_dsp.dsp.biquad as bq
import numpy as np


def _ws(locals):
    """
    without self.

    Parameters
    ----------
    locals : dict
        a dictionary

    Returns
    -------
    dict
        l with the entry "self" removed
    """
    return {k: v for k, v in locals.items() if k != "self"}


class Biquad(Stage):
    """
    A second order biquadratic filter, which can bse used to make many
    common second order filters. The filter is initialised in a
    bypass state, and the ``make_*`` methods can be used to calculate the
    coefficients.

    This Stage implements a direct form 1 biquad filter:
    ``a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]``

    For efficiency the biquad coefficients are normalised by ``a0`` and the
    output ``a`` coefficients multiplied by -1.

    Attributes
    ----------
    dsp_block : audio_dsp.dsp.biquad.biquad
        The dsp block class, see :class:`audio_dsp.dsp.biquad.biquad`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("biquad"), **kwargs)
        if self.fs is None:
            raise ValueError("Biquad requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.dsp_block: bq = bq.biquad_bypass(self.fs, self.n_in)
        self.set_control_field_cb(
            "filter_coeffs", lambda: [i for i in self._get_fixed_point_coeffs()]
        )
        self.set_control_field_cb("left_shift", lambda: self.dsp_block.b_shift)
        self.stage_memory_parameters = (self.n_in,)

    def _get_fixed_point_coeffs(self) -> np.ndarray:
        a = np.array(self.dsp_block.coeffs)
        return np.array(a * (2**30), dtype=np.int32)

    def make_bypass(self) -> "Biquad":
        """Make this biquad a bypass, by setting the b0 coefficient to
        1.
        """
        self.details = {}
        self.dsp_block = bq.biquad_bypass(self.fs, self.n_in)
        return self

    def make_lowpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad a second order low pass filter.

        Parameters
        ----------
        f : float
            cutoff frequency of the filter in Hz.
        q : float
            Q factor of the filter roll-off. 0.707 is equivalent to a
            Butterworth response.
        """
        self.details = dict(type="low pass", **_ws(locals()))
        self.dsp_block = bq.biquad_lowpass(self.fs, self.n_in, f, q)
        return self

    def make_highpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad a second order high pass filter.

        Parameters
        ----------
        f : float
            cutoff frequency of the filter in Hz.
        q : float
            Q factor of the filter roll-off. 0.707 is equivalent to a
            Butterworth response.
        """
        self.details = dict(type="high pass", **_ws(locals()))
        self.dsp_block = bq.biquad_highpass(self.fs, self.n_in, f, q)
        return self

    def make_bandpass(self, f: float, bw: float) -> "Biquad":
        """Make this biquad a second order bandpass filter.

        Parameters
        ----------
        f : float
            center frequency of the filter in Hz.
        bw : float
            Bandwidth of the filter in octaves.
        """
        self.details = dict(type="band pass", **_ws(locals()))
        self.dsp_block = bq.biquad_bandpass(self.fs, self.n_in, f, bw)
        return self

    def make_bandstop(self, f: float, bw: float) -> "Biquad":
        """Make this biquad a second order bandstop filter.

        Parameters
        ----------
        f : float
            center frequency of the filter in Hz.
        bw : float
            Bandwidth of the filter in octaves.
        """
        self.details = dict(type="band stop", **_ws(locals()))
        self.dsp_block = bq.biquad_bandstop(self.fs, self.n_in, f, bw)
        return self

    def make_notch(self, f: float, q: float) -> "Biquad":
        """Make this biquad a notch filter.

        Parameters
        ----------
        f : float
            center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        """
        self.details = dict(type="notch", **_ws(locals()))
        self.dsp_block = bq.biquad_notch(self.fs, self.n_in, f, q)
        return self

    def make_allpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad an all pass filter.

        Parameters
        ----------
        f : float
            center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        """
        self.details = dict(type="all pass", **_ws(locals()))
        self.dsp_block = bq.biquad_allpass(self.fs, self.n_in, f, q)
        return self

    def make_peaking(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a peaking filter.

        Parameters
        ----------
        f : float
            center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        self.details = dict(type="peaking", **_ws(locals()))
        self.dsp_block = bq.biquad_peaking(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_constant_q(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a peaking filter with constant q.

        Constant Q means that the bandwidth of the filter remains
        constant as the gain varies. It is commonly used for graphic
        equalisers.

        Parameters
        ----------
        f : float
            center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        self.details = dict(type="constant q", **_ws(locals()))
        self.dsp_block = bq.biquad_constant_q(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_lowshelf(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a second order low shelf filter.

        The Q factor is defined in a similar way to standard low pass,
        i.e. > 0.707 will yield peakiness (where the shelf response does
        not monotonically change). The level change at f will be
        boost_db/2.

        Parameters
        ----------
        f : float
            Cutoff frequency of the shelf in Hz, where the gain is
            boost_db/2
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        self.details = dict(type="lowshelf", **_ws(locals()))
        self.dsp_block = bq.biquad_lowshelf(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_highshelf(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a second order high shelf filter.

        The Q factor is defined in a similar way to standard high pass,
        i.e. > 0.707 will yield peakiness (where the shelf response does
        not monotonically change). The level change at f will be
        boost_db/2.

        Parameters
        ----------
        f : float
            Cutoff frequency of the shelf in Hz, where the gain is
            boost_db/2
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        self.details = dict(type="highshelf", **_ws(locals()))
        self.dsp_block = bq.biquad_highshelf(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_linkwitz(self, f0: float, q0: float, fp: float, qp: float) -> "Biquad":
        """Make this biquad a Linkwitz Transform biquad filter.

        The Linkwitz Transform is commonly used to change the low
        frequency roll off slope of a loudspeaker. When applied to a
        loudspeaker, it will change the cutoff frequency from f0 to fp,
        and the quality factor from q0 to qp.

        Parameters
        ----------
        f0 : float
            The original cutoff frequency of the filter in Hz.
        q0 : float
            The original quality factor of the filter at f0.
        fp : float
            The target cutoff frequency for the filter in Hz.
        qp : float
            The target quality factor for the filter.
        """
        self.details = dict(type="linkwitz", **_ws(locals()))
        self.dsp_block = bq.biquad_linkwitz(self.fs, self.n_in, f0, q0, fp, qp)
        return self
