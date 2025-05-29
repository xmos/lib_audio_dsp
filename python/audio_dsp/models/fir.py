from audio_dsp.models.stage import StageModel, StageConfig
from pydantic import Field, field_validator, model_validator
from typing import Literal
from pathlib import Path


class FirConfig(StageConfig):
    """Compile time configuration for a FIR Stage."""

    coeffs_path: Path = Field(default=Path(""), description="Path to filter coefficients file.")


class FirDirect(StageModel):
    """FIR filter stage using direct form implementation."""

    op_type: Literal["FirDirect"] = "FirDirect"
    config: FirConfig = Field(default_factory=FirConfig)
