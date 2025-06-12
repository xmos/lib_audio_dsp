# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Models for biquad filter stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.fields import BIQUAD_TYPES, biquad_bypass


class BiquadParameters(StageParameters):
    """Parameters for a biquad filter.

    Attributes
    ----------
        filter_type: The type of biquad filter to use (e.g., lowpass, highpass, etc.)
        slew_rate: Maximum rate of change for filter coefficients (units/sample)
    """

    filter_type: BIQUAD_TYPES = Field(
        default=biquad_bypass(), description="Type of biquad filter to implement"
    )


class Biquad(StageModel):
    """A single biquad filter stage.

    A biquad filter is a second-order recursive filter that can implement various
    filter types like lowpass, highpass, bandpass, etc. This stage implements a
    single biquad section with slew rate limiting to prevent audio artifacts
    when parameters are changed.
    """

    op_type: Literal["Biquad"] = "Biquad"
    parameters: BiquadParameters = Field(
        default_factory=lambda: BiquadParameters(filter_type=biquad_bypass())
    )


class BiquadSlewParameters(BiquadParameters):
    """Parameters for a slewing biquad filter."""

    slew_rate: float = Field(
        default=1.0, gt=0, description="Maximum rate of change for filter coefficients per sample"
    )


class BiquadSlew(StageModel):
    """A single biquad filter stage.

    A biquad filter is a second-order recursive filter that can implement various
    filter types like lowpass, highpass, bandpass, etc. This stage implements a
    single biquad section with slew rate limiting to prevent audio artifacts
    when parameters are changed.
    """

    op_type: Literal["BiquadSlew"] = "BiquadSlew"
    parameters: BiquadParameters = Field(
        default_factory=lambda: BiquadParameters(filter_type=biquad_bypass())
    )
