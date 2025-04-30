"""Models for biquad filter stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.biquad import BIQUAD_TYPES


class BiquadParameters(StageParameters):
    """Parameters for a biquad filter.

    Attributes
    ----------
        filter_type: The type of biquad filter to use (e.g., lowpass, highpass, etc.)
        slew_rate: Maximum rate of change for filter coefficients (units/sample)
    """

    filter_type: BIQUAD_TYPES = Field(..., description="Type of biquad filter to implement")
    slew_rate: float = Field(
        default=1.0, gt=0, description="Maximum rate of change for filter coefficients per sample"
    )


class Biquad(StageModel):
    """A single biquad filter stage.

    A biquad filter is a second-order recursive filter that can implement various
    filter types like lowpass, highpass, bandpass, etc. This stage implements a
    single biquad section with slew rate limiting to prevent audio artifacts
    when parameters are changed.
    """

    op_type: Literal["Biquad"] = "Biquad"
    parameters: BiquadParameters = Field(default_factory=BiquadParameters)
