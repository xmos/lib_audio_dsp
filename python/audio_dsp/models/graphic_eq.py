"""Pydantic models for signal chain DSP Stages."""

from typing import Literal, Annotated
from annotated_types import Len

from pydantic import Field, field_validator, model_validator

from audio_dsp.models.stage import NodePlacement, StageConfig, StageModel, StageParameters, Placement_2i1o, Placement_Ni1o, Placement_4i2o
from audio_dsp.models.fields import DEFAULT_GAIN_DB



class GraphicEq10bParameters(StageParameters):
    """Parameters for VolumeControl Stage."""

    gains_db: Annotated[
        list[Annotated[float, Field(ge=-24, le=24, description="Gain of the band in dB.")]],
        Len(10)
    ] = Field(default_factory=lambda: [0.0]*10)


class GraphicEq10b(StageModel):
    """
    This stage implements a volume control. The input signal is
    multiplied by a gain. The gain can be changed at runtime. To avoid
    pops and clicks during gain changes, a slew is applied to the gain
    update. The stage can be muted and unmuted at runtime.
    """

    op_type: Literal["GraphicEq10b"] = "GraphicEq10b"
    parameters: GraphicEq10bParameters = Field(default_factory=GraphicEq10bParameters)
