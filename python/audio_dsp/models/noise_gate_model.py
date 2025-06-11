"""Models for noise gate stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.fields import DEFAULT_ATTACK_T, DEFAULT_RELEASE_T, DEFAULT_COMPRESSOR_RATIO
from audio_dsp.models.fields import DEFAULT_THRESHOLD_DB


class NoiseGateParameters(StageParameters):
    """Parameters for noise gate stage.

    Attributes
    ----------
        threshold_db: Level below which the gate begins to close (dB)
        attack_t: Time for gate to open when signal exceeds threshold (seconds)
        release_t: Time for gate to close when signal falls below threshold (seconds)
    """

    threshold_db: float = DEFAULT_THRESHOLD_DB(
        description="Level in dB below which the gate begins to close",
    )
    attack_t: float = DEFAULT_ATTACK_T(
        description="Time in seconds for gate to open when signal exceeds threshold",
    )
    release_t: float = DEFAULT_RELEASE_T(
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
