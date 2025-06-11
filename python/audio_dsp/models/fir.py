# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""FIR model definitions."""

from audio_dsp.models.stage import StageModel, StageConfig
from pydantic import Field, field_validator, model_validator
from typing import Literal
from pathlib import Path


class FirConfig(StageConfig):
    """Compile time configuration for a FIR Stage."""

    coeffs_path: Path = Field(description="Path to filter coefficients file.")


class FirDirect(StageModel):
    """FIR filter stage using direct form implementation."""

    op_type: Literal["FirDirect"] = "FirDirect"
    config: FirConfig  # pyright: ignore Required field, no default or Field(...) needed 
