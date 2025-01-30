from typing import Annotated, Literal

from annotated_types import Len
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.models.biquad as bq

from .stage import StageConfig, StageModel, StageParameters


class CascadedBiquads(StageModel):
    """8 cascaded biquad filters. This allows up to 8 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
    - a parametric EQ, where several biquad filters are used at once.

    For documentation on the individual biquad filters, see
    :class:`audio_dsp.stages.biquad.Biquad` and
    :class:`audio_dsp.dsp.biquad.biquad`

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.cascaded_biquad.cascaded_biquad`
        The DSP block class; see :ref:`CascadedBiquads` for
        implementation details.

    """

    # class Model(Stage.Model):
    op_type: Literal["CascadedBiquads"] = "CascadedBiquads"


def _8biquads():
    return [bq.biquad_bypass() for _ in range(8)]


class ParametricEqParameters(StageParameters):
    filters: Annotated[list[bq.BIQUAD_TYPES], Len(8)] = Field(
        default_factory=_8biquads, max_items=8
    )


class ParametricEq(CascadedBiquads):
    # class Model(Stage.Model):
    op_type: Literal["ParametricEq"] = "ParametricEq"
    parameters: SkipJsonSchema[ParametricEqParameters] = Field(
        default_factory=ParametricEqParameters
    )
