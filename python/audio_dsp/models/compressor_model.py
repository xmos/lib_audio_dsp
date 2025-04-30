"""Models for compressor stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters


class CompressorParameters(StageParameters):
    """Parameters for compressor stage.

    Attributes
    ----------
        ratio: Compression ratio applied when signal exceeds threshold
        threshold_db: Level in dB above which compression occurs
        attack_t: Time for compressor to start compressing (seconds)
        release_t: Time for signal to return to original level (seconds)
    """

    ratio: float = Field(
        default=4.0,
        gt=1.0,
        le=20.0,
        description="Compression ratio applied when signal exceeds threshold",
    )
    threshold_db: float = Field(
        default=0.0, ge=-96.0, le=0.0, description="Level in dB above which compression occurs"
    )
    attack_t: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Time in seconds for compressor to start compressing",
    )
    release_t: float = Field(
        default=0.2,
        gt=0.0,
        le=5.0,
        description="Time in seconds for signal to return to original level",
    )


class CompressorRMS(StageModel):
    """Compressor stage based on RMS envelope of input signal.

    When the RMS envelope of the signal exceeds the threshold, the signal
    amplitude is reduced by the compression ratio. The threshold sets the value
    above which compression occurs. The ratio sets how much the signal is
    compressed. A ratio of 1 results in no compression, while a ratio of
    infinity results in the same behavior as a limiter.

    The attack time sets how fast the compressor starts compressing. The release
    time sets how long the signal takes to ramp up to its original level after
    the envelope is below the threshold.
    """

    op_type: Literal["CompressorRMS"] = "CompressorRMS"
    parameters: CompressorParameters = Field(default_factory=CompressorParameters)
