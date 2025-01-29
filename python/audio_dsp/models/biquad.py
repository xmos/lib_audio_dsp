from .stage import StageParameters
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