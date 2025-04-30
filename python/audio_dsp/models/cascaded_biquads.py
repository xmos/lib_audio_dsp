from typing import Annotated, Literal

from annotated_types import Len
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.models.biquad as bq

from .stage import StageConfig, StageModel, StageParameters


def _8biquads():
    return [bq.biquad_bypass() for _ in range(8)]

def _16biquads():
    return [bq.biquad_bypass() for _ in range(16)]

class CascadedBiquadParameters(StageParameters):
    filters: Annotated[list[bq.BIQUAD_TYPES], Len(8)] = Field(
        default_factory=_8biquads, max_length=8
    )


class CascadedBiquads(StageModel):
    """8 cascaded biquad filters. This allows up to 8 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
    - a parametric EQ, where several biquad filters are used at once.
    """

    # class Model(Stage.Model):
    op_type: Literal["CascadedBiquads"] = "CascadedBiquads"
    parameters: CascadedBiquadParameters = Field(
        default_factory=CascadedBiquadParameters
    )


class CascadedBiquad16Parameters(StageParameters):
    filters: Annotated[list[bq.BIQUAD_TYPES], Len(16)] = Field(
        default_factory=_8biquads, max_length=16
    )


class ParametricEq(CascadedBiquads):
    # class Model(Stage.Model):
    op_type: Literal["ParametricEq"] = "ParametricEq"
