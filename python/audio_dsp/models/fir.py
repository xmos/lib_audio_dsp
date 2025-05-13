from audio_dsp.models.stage import StageModel, StageConfig
from pydantic import Field, field_validator, model_validator
from typing import Literal


class FirConfig(StageConfig):
    """Compile time configuration for a FIR Stage."""

    coefficients: list[int] = Field(default=[], description="List of filter coefficients.", min_length=1)


class FirDirect(StageModel):
    """
    """

    op_type: Literal["FirDirect"] = "FirDirect"
    config: FirConfig = Field(default_factory=FirConfig)
