"""Models for sidechain compressor stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters, NodePlacement


class CompressorSidechainParameters(StageParameters):
    """Parameters for sidechain compressor stage.

    Attributes
    ----------
        ratio: Compression ratio applied when detect signal exceeds threshold
        threshold_db: Level in dB above which compression occurs
        attack_t: Time for compressor to start compressing (seconds)
        release_t: Time for signal to return to original level (seconds)
    """

    ratio: float = Field(
        default=3.0,
        gt=1.0,
        le=20.0,
        description="Compression ratio applied when detect signal exceeds threshold",
    )
    threshold_db: float = Field(
        default=-35.0, ge=-96.0, le=0.0, description="Level in dB above which compression occurs"
    )
    attack_t: float = Field(
        default=0.005,
        gt=0.0,
        le=1.0,
        description="Time in seconds for compressor to start compressing",
    )
    release_t: float = Field(
        default=0.120,
        gt=0.0,
        le=5.0,
        description="Time in seconds for signal to return to original level",
    )


class CompressorSidechainPlacement(NodePlacement):
    """Node placement for sidechain compressor.

    Requires exactly 2 inputs:
    - Input 0: Signal to be compressed
    - Input 1: Detect signal used to control compression

    Produces exactly 1 output:
    - Output 0: Compressed version of input 0
    """

    input: list[int] = Field(
        default=[],
        description="List of input edges.",
        min_length=2,
        max_length=2,
    )
    output: list[int] = Field(
        default=[], description="IDs of output edges.", min_length=1, max_length=1
    )


class CompressorSidechain(StageModel[CompressorSidechainPlacement]):
    """Sidechain compressor stage based on RMS envelope of detect signal.

    This stage requires exactly 2 input channels:
    1. The signal to be compressed
    2. The detect signal that controls the compression

    When the RMS envelope of the detect signal exceeds the threshold, the
    processed signal amplitude is reduced by the compression ratio. The threshold
    sets the value above which compression occurs. The ratio sets how much the
    signal is compressed. A ratio of 1 results in no compression, while a ratio
    of infinity results in the same behavior as a limiter.

    The attack time sets how fast the compressor starts compressing. The release
    time sets how long the signal takes to ramp up to its original level after
    the envelope is below the threshold.
    """

    op_type: Literal["CompressorSidechain"] = "CompressorSidechain"
    parameters: CompressorSidechainParameters = Field(
        default_factory=CompressorSidechainParameters
    )
