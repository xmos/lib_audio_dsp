# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Biquad Stages can be used for basic audio filters."""

from audio_dsp.design.stage import Stage, find_config, StageParameters
import audio_dsp.dsp.biquad as bq
from typing import Any, Literal
from functools import partial
from pydantic import BaseModel, RootModel, Field, create_model
from typing import Literal, Annotated, List, Union
from annotated_types import Len

def _ws(locals):
    """
    Without self.

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


DEFAULT_Q = partial(Field, 0.707, gt=0, le=10, description="Q factor of the filter.")
DEFAULT_FILTER_FREQ = partial(
    Field, 500, gt=0, lt=24000, description="Frequency of the filter in Hz."
)  # 48kHz sample rate
DEFAULT_BW = partial(
    Field, 1, gt=0, le=10, description="Bandwidth of the filter in octaves."
)
DEFAULT_BOOST_DB = partial(
    Field, 0, ge=-24, le=24, description="Gain of the filter in dB."
)

class biquad_allpass(StageParameters):
    type: Literal["allpass"] = "allpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_bandpass(StageParameters):
    type: Literal["bandpass"] = "bandpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    bw: float = DEFAULT_BW()


class biquad_bandstop(StageParameters):
    type: Literal["bandstop"] = "bandstop"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    bw: float = DEFAULT_BW()


class biquad_bypass(StageParameters):
    type: Literal["bypass"] = "bypass"


class biquad_constant_q(StageParameters):
    type: Literal["constant_q"] = "constant_q"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


class biquad_gain(StageParameters):
    type: Literal["gain"] = "gain"
    gain_db: float = 0


class biquad_highpass(StageParameters):
    type: Literal["highpass"] = "highpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_highshelf(StageParameters):
    type: Literal["highshelf"] = "highshelf"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


class biquad_linkwitz(StageParameters):
    type: Literal["linkwitz"] = "linkwitz"
    f0: float = 500
    q0: float = DEFAULT_Q()
    fp: float = 1000
    qp: float = DEFAULT_Q()


class biquad_lowpass(StageParameters):
    type: Literal["lowpass"] = "lowpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_lowshelf(StageParameters):
    type: Literal["lowshelf"] = "lowshelf"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


class biquad_notch(StageParameters):
    type: Literal["notch"] = "notch"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_peaking(StageParameters):
    type: Literal["peaking"] = "peaking"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


BIQUAD_TYPES = Union[
    biquad_allpass,
    biquad_bandpass,
    biquad_bandstop,
    biquad_bypass,
    biquad_constant_q,
    biquad_gain,
    biquad_highpass,
    biquad_highshelf,
    biquad_linkwitz,
    biquad_lowpass,
    biquad_lowshelf,
    biquad_notch,
    biquad_peaking,
]

class Biquad(Stage):
    """
    A second order biquadratic filter, which can be used to make many
    common second order filters. The filter is initialised in a
    bypass state, and the ``make_*`` methods can be used to calculate the
    coefficients.

    This Stage implements a direct form 1 biquad filter:
    ``a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]``

    For efficiency the biquad coefficients are normalised by ``a0`` and the
    output ``a`` coefficients multiplied by -1.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.biquad.biquad`
        The DSP block class; see :ref:`Biquad`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("biquad"), **kwargs)
        if self.fs is None:
            raise ValueError("Biquad requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.dsp_block: bq = bq.biquad_bypass(self.fs, self.n_in)
        self.set_control_field_cb("filter_coeffs", self._get_fixed_point_coeffs)
        self.set_control_field_cb("left_shift", lambda: self.dsp_block.b_shift)
        self.stage_memory_parameters = (self.n_in,)

    def _get_fixed_point_coeffs(self) -> list[int]:
        return self.dsp_block.int_coeffs

    def make_bypass(self) -> "Biquad":
        """Make this biquad a bypass by setting the b0 coefficient to
        1.
        """
        self.details = {}
        new_coeffs = bq.make_biquad_bypass(self.fs)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_lowpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad a second order low pass filter.

        Parameters
        ----------
        f : float
            Cutoff frequency of the filter in Hz.
        q : float
            Q factor of the filter roll-off. 0.707 is equivalent to a
            Butterworth response.
        """
        self.details = dict(type="low pass", **_ws(locals()))
        new_coeffs = bq.make_biquad_lowpass(self.fs, f, q)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_highpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad a second order high pass filter.

        Parameters
        ----------
        f : float
            Cutoff frequency of the filter in Hz.
        q : float
            Q factor of the filter roll-off. 0.707 is equivalent to a
            Butterworth response.
        """
        self.details = dict(type="high pass", **_ws(locals()))
        new_coeffs = bq.make_biquad_highpass(self.fs, f, q)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_bandpass(self, f: float, bw: float) -> "Biquad":
        """Make this biquad a second order bandpass filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        bw : float
            Bandwidth of the filter in octaves.
        """
        self.details = dict(type="band pass", **_ws(locals()))
        new_coeffs = bq.make_biquad_bandpass(self.fs, f, bw)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_bandstop(self, f: float, bw: float) -> "Biquad":
        """Make this biquad a second order bandstop filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        bw : float
            Bandwidth of the filter in octaves.
        """
        self.details = dict(type="band stop", **_ws(locals()))
        new_coeffs = bq.make_biquad_bandstop(self.fs, f, bw)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_notch(self, f: float, q: float) -> "Biquad":
        """Make this biquad a notch filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        """
        self.details = dict(type="notch", **_ws(locals()))
        new_coeffs = bq.make_biquad_notch(self.fs, f, q)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_allpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad an all pass filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        """
        self.details = dict(type="all pass", **_ws(locals()))
        new_coeffs = bq.make_biquad_allpass(self.fs, f, q)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_peaking(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a peaking filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        self.details = dict(type="peaking", **_ws(locals()))
        new_coeffs = bq.make_biquad_peaking(self.fs, f, q, boost_db)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_constant_q(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a peaking filter with constant Q.

        Constant Q means that the bandwidth of the filter remains
        constant as the gain varies. It is commonly used for graphic
        equalisers.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        self.details = dict(type="constant q", **_ws(locals()))
        new_coeffs = bq.make_biquad_constant_q(self.fs, f, q, boost_db)
        self.dsp_block.update_coeffs(new_coeffs)
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
        new_coeffs = bq.make_biquad_lowshelf(self.fs, f, q, boost_db)
        self.dsp_block.update_coeffs(new_coeffs)
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
        new_coeffs = bq.make_biquad_highshelf(self.fs, f, q, boost_db)
        self.dsp_block.update_coeffs(new_coeffs)
        return self

    def make_linkwitz(self, f0: float, q0: float, fp: float, qp: float) -> "Biquad":
        """Make this biquad a Linkwitz Transform biquad filter.

        The Linkwitz Transform changes the low frequency cutoff of a filter, and is commonly used to change the low
        frequency roll off slope of a loudspeaker. When applied to a
        loudspeaker, it will change the cutoff frequency from f0 to fp,
        and the Q factor from q0 to qp.

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
        new_coeffs = bq.make_biquad_linkwitz(self.fs, f0, q0, fp, qp)
        self.dsp_block.update_coeffs(new_coeffs)
        return self


class BiquadSlew(Biquad):
    """
    A second order biquadratic filter with slew, which can be used to
    make many common second order filters. The filter is initialised in a
    bypass state, and the ``make_*`` methods can be used to calculate the
    coefficients. This variant will slew between filter coefficients when
    they are changed.

    This Stage implements a direct form 1 biquad filter:
    ``a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]``

    For efficiency the biquad coefficients are normalised by ``a0`` and the
    output ``a`` coefficients multiplied by -1.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.biquad.biquad_slew`
        The DSP block class; see :ref:`Biquad`
        for implementation details.
    """

    def __init__(self, **kwargs):
        Stage.__init__(self, config=find_config("biquad_slew"), **kwargs)
        if self.fs is None:
            raise ValueError("Biquad slew requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        init_coeffs = bq.make_biquad_bypass(self.fs)
        self.dsp_block: bq = bq.biquad_slew(init_coeffs, self.fs, self.n_in)
        self.set_control_field_cb("filter_coeffs", self._get_fixed_point_coeffs)
        self.set_control_field_cb("left_shift", lambda: self.dsp_block.b_shift)
        self.set_control_field_cb("slew_shift", lambda: self.dsp_block.slew_shift)
        self.stage_memory_parameters = (self.n_in,)

    def _get_fixed_point_coeffs(self) -> list:
        return self.dsp_block.target_coeffs_int

    def set_slew_shift(self, slew_shift):
        """Set the slew shift for a biquad object. This sets how fast the
        filter will slew between filter coefficients.
        """
        self.dsp_block.slew_shift = slew_shift
