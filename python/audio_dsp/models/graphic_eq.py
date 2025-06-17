# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Pydantic models for signal chain DSP Stages."""

from typing import Literal, Annotated
from annotated_types import Len

from pydantic import Field, field_validator, model_validator

from audio_dsp.models.stage import (
    NodePlacement,
    StageConfig,
    StageModel,
    StageParameters,
    Placement_2i1o,
    Placement_Ni1o,
    Placement_4i2o,
)
from audio_dsp.models.fields import DEFAULT_GAIN_DB


GEQ_GAIN = Field(ge=-24, le=24, description="Gain of the band in dB.")


class GraphicEq10bParameters(StageParameters):
    """
    Parameters for a 10-band graphic equalizer.

    Attributes
    ----------
        gains_db: (list[float])
            Gain values (in dB) for each of the 10 frequency bands.
            - Each value must be between -24 dB and +24 dB.
            - The list must have exactly 10 elements.
    """

    gains_db: Annotated[
        list[Annotated[float, GEQ_GAIN]],
        Len(10),
    ] = Field(default_factory=lambda: [0.0] * 10)


class GraphicEq10b(StageModel):
    """
    This stage implements a Graphic EQ with 10 bands.
    Each band can be adjusted with a gain value ranging from -24 dB to +24 dB.
    """

    op_type: Literal["GraphicEq10b"] = "GraphicEq10b"
    parameters: GraphicEq10bParameters = Field(default_factory=GraphicEq10bParameters)
