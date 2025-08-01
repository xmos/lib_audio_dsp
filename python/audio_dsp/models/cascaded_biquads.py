# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Pydantic models of the cascaded biquad DSP Stages."""

from typing import Annotated, Literal

from annotated_types import Len
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from audio_dsp.models.fields import BIQUAD_TYPES, biquad_bypass, DEFAULT_FILTER_FREQ

from .stage import StageModel, StageParameters


def _8biquads():
    return [biquad_bypass() for _ in range(8)]


def _16biquads():
    return [biquad_bypass() for _ in range(16)]


CASCADED_BIQUADS_8 = Annotated[list[BIQUAD_TYPES], Len(8)]


class CascadedBiquadsParameters(StageParameters):
    """Parameters for CascadedBiquad Stage.

    Attributes
    ----------
    filters : list[BIQUAD_TYPES]
        A list of BiquadParameters to update the cascaded biquads with.
    """

    filters: CASCADED_BIQUADS_8 = Field(default_factory=_8biquads, max_length=8)


class NthOrderFilterParameters(StageParameters):
    """Parameters for NthOrderFilter Stage."""

    type: Literal["bypass", "highpass", "lowpass"] = Field(
        default="bypass",
        description="Type of filter to implement. Can be 'bypass', 'highpass', or 'lowpass'.",
    )
    filter: Literal["butterworth"] = Field(
        default="butterworth",
        description="Class of filter to use. Currently only 'butterworth' is supported.",
    )
    order: Literal[2, 4, 6, 8, 10, 12, 14, 16] = Field(
        default=2, description="The order of the filter. Must be even and less than 16."
    )
    filter_freq: float = DEFAULT_FILTER_FREQ(
        description="-3dB cutoff frequency of the filter in Hz."
    )


class CascadedBiquads(StageModel):
    """8 cascaded biquad filters. This allows up to 8 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
    - a parametric EQ, where several biquad filters are used at once.
    """

    op_type: Literal["CascadedBiquads"] = "CascadedBiquads"
    parameters: CascadedBiquadsParameters | NthOrderFilterParameters = Field(
        default=CascadedBiquadsParameters()
    )


class CascadedBiquads16Parameters(StageParameters):
    """Parameters for CascadedBiquad16 Stage."""

    filters: Annotated[list[BIQUAD_TYPES], Len(16)] = Field(
        default_factory=_16biquads, max_length=16
    )


class CascadedBiquads16(StageModel):
    """8 cascaded biquad filters. This allows up to 8 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
    - a parametric EQ, where several biquad filters are used at once.
    """

    op_type: Literal["CascadedBiquads16"] = "CascadedBiquads16"
    parameters: CascadedBiquads16Parameters = Field(default_factory=CascadedBiquads16Parameters)


class ParametricEq8b(CascadedBiquads):
    """Pydantic model of the ParametricEq8b Stage."""

    op_type: Literal["ParametricEq8b"] = "ParametricEq8b"  # pyright: ignore override


class ParametricEq16b(CascadedBiquads16):
    """Pydantic model of the ParametricEq16b Stage."""

    op_type: Literal["ParametricEq16b"] = "ParametricEq16b"  # pyright: ignore override


class NthOrderFilter(StageModel):
    """8 cascaded biquad filters. This allows up to 8 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
    - a parametric EQ, where several biquad filters are used at once.
    """

    op_type: Literal["NthOrderFilter"] = "NthOrderFilter"
    parameters: NthOrderFilterParameters = Field(default_factory=NthOrderFilterParameters)
