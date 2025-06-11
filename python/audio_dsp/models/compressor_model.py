"""Models for compressor stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters
from audio_dsp.models.fields import DEFAULT_ATTACK_T, DEFAULT_RELEASE_T, DEFAULT_COMPRESSOR_RATIO
from audio_dsp.models.fields import DEFAULT_THRESHOLD_DB


class CompressorParameters(StageParameters):
    """Parameters for compressor stage."""

    ratio: float = DEFAULT_COMPRESSOR_RATIO(
        description="Compression ratio applied when detect signal exceeds threshold"
    )
    threshold_db: float = DEFAULT_THRESHOLD_DB(
        description="Level in dB above which compression occurs"
    )
    attack_t: float = DEFAULT_ATTACK_T(
        description="Time in seconds for compressor to start compressing"
    )
    release_t: float = DEFAULT_RELEASE_T(
        description="Time in seconds for signal to return to original level"
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
