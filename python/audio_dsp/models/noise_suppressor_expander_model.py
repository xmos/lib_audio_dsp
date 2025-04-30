"""Models for noise suppressor expander stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters


class NoiseSuppressorExpanderParameters(StageParameters):
    """Parameters for noise suppressor expander stage.

    Attributes
    ----------
        ratio: Expansion ratio applied when signal falls below threshold
        threshold_db: Level in dB below which expansion occurs
        attack_t: Time for expander to start expanding (seconds)
        release_t: Time for signal to return to original level (seconds)
    """

    ratio: float = Field(
        default=3.0,
        gt=1.0,
        le=20.0,
        description="Expansion ratio applied when signal falls below threshold",
    )
    threshold_db: float = Field(
        default=-35.0, ge=-96.0, le=0.0, description="Level in dB below which expansion occurs"
    )
    attack_t: float = Field(
        default=0.005,
        gt=0.0,
        le=1.0,
        description="Time in seconds for expander to start expanding",
    )
    release_t: float = Field(
        default=0.120,
        gt=0.0,
        le=5.0,
        description="Time in seconds for signal to return to original level",
    )


class NoiseSuppressorExpander(StageModel):
    """Noise suppressor expander stage.

    A noise suppressor that reduces the level of an audio signal when it falls
    below a threshold. This is also known as an expander. When the signal
    envelope falls below the threshold, the gain applied to the signal is reduced
    relative to the expansion ratio over the release time.

    When the envelope returns above the threshold, the gain applied to the signal
    is increased to 1 over the attack time. The initial state of the noise
    suppressor is with the suppression off, modeling a full scale signal having
    been present before t = 0.
    """

    op_type: Literal["NoiseSuppressorExpander"] = "NoiseSuppressorExpander"
    parameters: NoiseSuppressorExpanderParameters = Field(
        default_factory=NoiseSuppressorExpanderParameters
    )
