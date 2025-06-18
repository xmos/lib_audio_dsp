# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Pydantic models of the different biquad types."""

from audio_dsp.models.stage import StageParameters
from functools import partial
from pydantic import BaseModel, RootModel, Field, create_model
from typing import Literal, Annotated, List, Union
from annotated_types import Len
from audio_dsp.dsp.generic import HEADROOM_DB, MIN_SIG_DB, Q_SIG
from audio_dsp.dsp import utils


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


DEFAULT_Q = partial(Field, default=0.707, gt=0, le=10, description="Q factor of the filter.")
DEFAULT_FILTER_FREQ = partial(
    Field, default=1000, gt=0, lt=24000, description="Frequency of the filter in Hz."
)
DEFAULT_BW = partial(
    Field, default=1, gt=0, le=10, description="Bandwidth of the filter in octaves."
)
DEFAULT_BOOST_DB = partial(
    Field, default=0.0, ge=-24, le=24, description="Gain of the filter in dB."
)

DEFAULT_GAIN_DB = partial(
    Field, default=0.0, ge=MIN_SIG_DB, le=HEADROOM_DB, description="Gain of the stage in dB."
)

DEFAULT_ATTACK_T = partial(
    Field, default=0.01, gt=0, le=1, description="Attack time of the stage in seconds."
)
DEFAULT_RELEASE_T = partial(
    Field, default=0.2, gt=0, le=5, description="Release time of the stage in seconds."
)
DEFAULT_COMPRESSOR_RATIO = partial(
    Field, default=4.0, gt=1, le=20, description="Compression ratio of the stage."
)
DEFAULT_THRESHOLD_DB = DEFAULT_GAIN_DB

RMS_HEADROOM_DB = utils.db_pow((utils.Q_max(31) + 1) / utils.Q_max(Q_SIG))
MIN_RMS_SIG_DB = utils.db_pow(1 / 2**Q_SIG)

DEFAULT_RMS_THRESHOLD_DB = partial(
    Field,
    default=0.0,
    ge=MIN_RMS_SIG_DB,
    le=RMS_HEADROOM_DB,
    description="Threshold of the stage in dB.",
)


class biquad_allpass(StageParameters):
    """Parameters for a Biquad Stage configured to allpass."""

    type: Literal["allpass"] = "allpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_bandpass(StageParameters):
    """Parameters for a Biquad Stage configured to bandpass."""

    type: Literal["bandpass"] = "bandpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    bw: float = DEFAULT_BW()


class biquad_bandstop(StageParameters):
    """Parameters for a Biquad Stage configured to bandstop."""

    type: Literal["bandstop"] = "bandstop"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    bw: float = DEFAULT_BW()


class biquad_bypass(StageParameters):
    """Parameters for a Biquad Stage configured to bypass."""

    type: Literal["bypass"] = "bypass"


class biquad_constant_q(StageParameters):
    """Parameters for a Biquad Stage configured to constant_q."""

    type: Literal["constant_q"] = "constant_q"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


class biquad_gain(StageParameters):
    """Parameters for a Biquad Stage configured to gain."""

    type: Literal["gain"] = "gain"
    gain_db: float = 0


class biquad_highpass(StageParameters):
    """Parameters for a Biquad Stage configured to highpass."""

    type: Literal["highpass"] = "highpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_highshelf(StageParameters):
    """Parameters for a Biquad Stage configured to highshelf."""

    type: Literal["highshelf"] = "highshelf"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


class biquad_linkwitz(StageParameters):
    """Parameters for a Biquad Stage configured to linkwitz."""

    type: Literal["linkwitz"] = "linkwitz"
    f0: float = DEFAULT_FILTER_FREQ()
    q0: float = DEFAULT_Q()
    fp: float = DEFAULT_FILTER_FREQ()
    qp: float = DEFAULT_Q()


class biquad_lowpass(StageParameters):
    """Parameters for a Biquad Stage configured to lowpass."""

    type: Literal["lowpass"] = "lowpass"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_lowshelf(StageParameters):
    """Parameters for a Biquad Stage configured to lowshelf."""

    type: Literal["lowshelf"] = "lowshelf"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()
    boost_db: float = DEFAULT_BOOST_DB()


class biquad_mute(StageParameters):
    """Parameters for a Biquad Stage configured to mute."""

    type: Literal["mute"] = "mute"


class biquad_notch(StageParameters):
    """Parameters for a Biquad Stage configured to notch."""

    type: Literal["notch"] = "notch"
    filter_freq: float = DEFAULT_FILTER_FREQ()
    q_factor: float = DEFAULT_Q()


class biquad_peaking(StageParameters):
    """Parameters for a Biquad Stage configured to peaking."""

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
    biquad_mute,
    biquad_notch,
    biquad_peaking,
]
