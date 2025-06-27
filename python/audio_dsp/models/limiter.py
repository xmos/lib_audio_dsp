# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Models for limiter stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.fields import (
    DEFAULT_ATTACK_T,
    DEFAULT_RELEASE_T,
    DEFAULT_THRESHOLD_DB,
    DEFAULT_RMS_THRESHOLD_DB,
)


class LimiterParameters(StageParameters):
    """Parameters for limiter stage."""

    threshold_db: float = DEFAULT_THRESHOLD_DB(
        description="Level in dB above which limiting occurs"
    )
    attack_t: float = DEFAULT_ATTACK_T(description="Time in seconds for limiter to start limiting")
    release_t: float = DEFAULT_RELEASE_T(
        description="Time in seconds for signal to return to original level"
    )


class LimiterRMSParameters(LimiterParameters):
    """Parameters for RMS limiter stage."""

    threshold_db: float = DEFAULT_RMS_THRESHOLD_DB(
        description="Level in dB above which limiting occurs"
    )


class LimiterRMS(StageModel):
    """Limiter stage based on RMS value of signal.

    When the RMS envelope of the signal exceeds the threshold, the signal
    amplitude is reduced. The threshold sets the value above which limiting
    occurs.

    The attack time sets how fast the limiter starts limiting. The release
    time sets how long the signal takes to ramp up to its original level after
    the envelope is below the threshold.
    """

    op_type: Literal["LimiterRMS"] = "LimiterRMS"
    parameters: LimiterRMSParameters = Field(default_factory=LimiterRMSParameters)


class LimiterPeak(StageModel):
    """Limiter stage based on peak value of signal.

    When the peak envelope of the signal exceeds the threshold, the signal
    amplitude is reduced. The threshold sets the value above which limiting
    occurs.

    The attack time sets how fast the limiter starts limiting. The release
    time sets how long the signal takes to ramp up to its original level after
    the envelope is below the threshold.
    """

    op_type: Literal["LimiterPeak"] = "LimiterPeak"
    parameters: LimiterParameters = Field(default_factory=LimiterParameters)


class HardLimiterPeak(StageModel):
    """Hard limiter stage based on peak value of signal.

    When the peak envelope of the signal exceeds the threshold, the signal
    amplitude is reduced. If the signal still exceeds the threshold, it is
    clipped. The peak envelope of the signal may never exceed the threshold.

    The threshold sets the value above which limiting/clipping occurs. The
    attack time sets how fast the limiter starts limiting. The release time
    sets how long the signal takes to ramp up to its original level after
    the envelope is below the threshold.
    """

    op_type: Literal["HardLimiterPeak"] = "HardLimiterPeak"
    parameters: LimiterParameters = Field(default_factory=LimiterParameters)


class ClipperParameters(StageParameters):
    """Parameters for clipper stage."""

    threshold_db: float = DEFAULT_THRESHOLD_DB(
        description="Level in dB above which clipping occurs"
    )


class Clipper(StageModel):
    """Clipper stage model for limiting signal amplitude."""

    op_type: Literal["Clipper"] = "Clipper"
    parameters: ClipperParameters = Field(default_factory=ClipperParameters)
