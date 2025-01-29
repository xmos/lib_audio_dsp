

from .stage import StageModel, StageParameters, StageConfig
from typing import Literal
from pydantic import Field
import audio_dsp.models.biquad as bq
from typing import Literal, Annotated, List, Union
from annotated_types import Len

class CascadedBiquadsModel(StageModel):
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
    default_factory=_8biquads, max_items=8)

class ParametricEq(CascadedBiquadsModel):
    # class Model(Stage.Model):
    op_type: Literal["ParametricEq"] = "ParametricEq"
    parameters: ParametricEqParameters = Field(default_factory=ParametricEqParameters)

