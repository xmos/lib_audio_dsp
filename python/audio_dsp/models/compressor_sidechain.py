# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Models for sidechain compressor stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import (
    StageModel,
    StageParameters,
    NodePlacement,
    Placement_2i1o,
    Placement_4i2o,
)
from audio_dsp.models.fields import (
    DEFAULT_ATTACK_T,
    DEFAULT_RELEASE_T,
    DEFAULT_COMPRESSOR_RATIO,
    DEFAULT_THRESHOLD_DB,
    DEFAULT_RMS_THRESHOLD_DB,
)


class CompressorSidechainParameters(StageParameters):
    """Parameters for sidechain compressor stage."""

    ratio: float = DEFAULT_COMPRESSOR_RATIO(
        description="Compression ratio applied when detect signal exceeds threshold"
    )
    threshold_db: float = DEFAULT_RMS_THRESHOLD_DB(
        description="Level in dB above which compression occurs"
    )
    attack_t: float = DEFAULT_ATTACK_T(
        description="Time in seconds for compressor to start compressing"
    )
    release_t: float = DEFAULT_RELEASE_T(
        description="Time in seconds for signal to return to original level"
    )


class CompressorSidechainPlacement(Placement_2i1o):
    """Node placement for sidechain compressor.

    Requires exactly 2 inputs:
    - Input 0: Signal to be compressed
    - Input 1: Detect signal used to control compression

    Produces exactly 1 output:
    - Output 0: Compressed version of input 0
    """


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


class CompressorSidechainStereo(StageModel[Placement_4i2o]):
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

    op_type: Literal["CompressorSidechainStereo"] = "CompressorSidechainStereo"
    parameters: CompressorSidechainParameters = Field(
        default_factory=CompressorSidechainParameters
    )
