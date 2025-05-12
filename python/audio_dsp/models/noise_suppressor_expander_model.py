"""Models for noise suppressor expander stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.fields import (
    DEFAULT_ATTACK_T,
    DEFAULT_RELEASE_T,
    DEFAULT_COMPRESSOR_RATIO,
    DEFAULT_THRESHOLD_DB,
)

class NoiseSuppressorExpanderParameters(StageParameters):
    """Parameters for noise suppressor expander stage."""

    ratio: float = DEFAULT_COMPRESSOR_RATIO(
    description="Expansion ratio applied when signal falls below threshold",
    )
    threshold_db: float = DEFAULT_THRESHOLD_DB(
        default=-35.0, description="Level in dB below which expansion occurs"
    )
    attack_t: float = DEFAULT_ATTACK_T(
        description="Time in seconds for expander to start expanding",
    )
    release_t: float = DEFAULT_RELEASE_T(
        description="Time in seconds for signal to return to original level",
    )


class NoiseSuppressorExpander(StageModel):
    """Noise suppressor expander stage.

    A noise suppressor that reduces the level of an audio signal when it falls below a threshold. This is also known as an expander.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced relative to the expansion ratio over the
    release time. When the envelope returns above the threshold, the
    gain applied to the signal is increased to 1 over the attack time.

    The initial state of the noise suppressor is with the suppression
    off; this models a full scale signal having been present before
    t = 0.
    """

    op_type: Literal["NoiseSuppressorExpander"] = "NoiseSuppressorExpander"
    parameters: NoiseSuppressorExpanderParameters = Field(
        default_factory=NoiseSuppressorExpanderParameters
    )
