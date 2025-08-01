# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Models for noise gate stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.fields import DEFAULT_ATTACK_T, DEFAULT_RELEASE_T, DEFAULT_COMPRESSOR_RATIO
from audio_dsp.models.fields import DEFAULT_THRESHOLD_DB


class NoiseGateParameters(StageParameters):
    """Parameters for noise gate stage."""

    threshold_db: float = DEFAULT_THRESHOLD_DB(
        default=-35,
        description="Level in dB below which the gate begins to close",
    )
    attack_t: float = DEFAULT_ATTACK_T(
        default=0.005,
        description="Time in seconds for gate to open when signal exceeds threshold",
    )
    release_t: float = DEFAULT_RELEASE_T(
        default=0.12,
        description="Time in seconds for gate to close when signal falls below threshold",
    )


class NoiseGate(StageModel):
    """Noise gate stage for removing low-level signals.

    Attenuates signals that fall below a threshold, useful for removing
    background noise during silent passages. When the signal falls below
    the threshold, the gain is reduced to 0 over the release time. When
    the signal rises above the threshold, the gain is increased to 1 over
    the attack time.
    """

    op_type: Literal["NoiseGate"] = "NoiseGate"
    parameters: NoiseGateParameters = Field(default_factory=NoiseGateParameters)
