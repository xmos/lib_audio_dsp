"""Models for noise gate stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters


class NoiseGateParameters(StageParameters):
    """Parameters for noise gate stage.
    
    Attributes:
        threshold_db: Level below which the gate begins to close (dB)
        attack_t: Time for gate to open when signal exceeds threshold (seconds)
        release_t: Time for gate to close when signal falls below threshold (seconds)
    """
    threshold_db: float = Field(
        default=-40.0,
        ge=-96.0,
        le=0.0,
        description="Level in dB below which the gate begins to close"
    )
    attack_t: float = Field(
        default=0.010,
        gt=0.0,
        le=1.0,
        description="Time in seconds for gate to open when signal exceeds threshold"
    )
    release_t: float = Field(
        default=0.100,
        gt=0.0,
        le=5.0,
        description="Time in seconds for gate to close when signal falls below threshold"
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