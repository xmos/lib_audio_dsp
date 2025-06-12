# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Pydantic models for reverb DSP Stages."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from audio_dsp.models.stage import (
    StageConfig,
    StageModel,
    StageParameters,
    MonoPlacement,
    StereoPlacement,
    NodePlacement,
)


class ReverbBaseParameters(StageParameters):
    """Parameters for all Reverb Stages."""

    predelay: float = Field(
        default=15, ge=0, le=30, description="Set the predelay in milliseconds."
    )
    pregain: float = Field(
        default=0.015,
        ge=0,
        le=1,
        description="It is not advised to increase this value above the "
        "default 0.015, as it can result in saturation inside "
        "the reverb delay lines.",
    )
    wet_dry_mix: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "The mix between the wet and dry signal. When the mix is 0, the output signal is fully dry, "
            "when 1, the output signal is fully wet."
        ),
    )


class ReverbStereoBaseParameters(ReverbBaseParameters):
    """Parameters for all stereo Reverb Stages."""

    width: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description=(
            "How much stereo separation there is between the left and "
            "right channels. Setting width to 0 will yield a mono signal, "
            "whilst setting width to 1 will yield the most stereo "
            "separation."
        ),
    )


class ReverbBaseConfig(StageConfig):
    """Compile time configuration for a ReverbRoom Stage."""

    max_predelay: float = Field(
        default=30, description="Set the maximum predelay in milliseconds."
    )


class _ReverbBaseModel[Placement: NodePlacement](StageModel[Placement]):
    """
    The base class for reverb stages, containing pre delays, and wet/dry
    mixes and pregain.
    """

    # op_type: is not defined as this Stage cannot be pipelined


class ReverbRoomConfig(ReverbBaseConfig):
    """Compile time configuration for a ReverbRoom Stage."""

    max_room_size: float = Field(
        default=1.0,
        gt=0,
        le=4,
        description=(
            "Sets the maximum room size for this reverb. The"
            " ``room_size`` parameter sets the fraction of this value actually used at any given time."
            " For optimal memory usage, max_room_size should be set so that the longest reverb tail"
            " occurs when ``room_size=1.0``."
        ),
    )


class ReverbRoomParameters(ReverbBaseParameters):
    """Parameters for a ReverbRoom Stage."""

    damping: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="This controls how much high frequency attenuation "
        "is in the room. Higher values yield shorter "
        "reverberation times at high frequencies. Range: 0 to 1",
    )
    decay: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="This sets how reverberant the room is. Higher "
        "values will give a longer reverberation time for "
        "a given room size. Range: 0 to 1",
    )
    room_size: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="This sets how reverberant the room is. Higher "
        "values will give a longer reverberation time for "
        "a given room size. Range: 0 to 1",
    )


class ReverbRoomStereoConfig(ReverbRoomConfig):
    """Compile time configuration for a ReverbRoomStereo Stage."""

    pass


class ReverbRoomStereoParameters(ReverbStereoBaseParameters, ReverbRoomParameters):
    """Parameters for a ReverbRoomStereo Stage."""

    pass


class ReverbPlateStereoConfig(ReverbBaseConfig):
    """Compile time configuration for a ReverbPlateStereo Stage."""

    pass


class ReverbPlateStereoParameters(ReverbStereoBaseParameters):
    """Parameters for a ReverbPlateStereo Stage."""

    pregain: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="It is not advised to increase this value above the "
        "default 0.5, as it can result in saturation inside the reverb delay lines.",
    )
    damping: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="This controls how much high frequency attenuation is in the room. Higher "
        "values yield shorter reverberation times at high frequencies. Range: 0 to 1",
    )
    decay: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="This sets how reverberant the room is. Higher "
        "values will give a longer reverberation time for "
        "a given room size. Range: 0 to 1",
    )
    early_diffusion: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Sets how much diffusion is present in the first part of the reverberation. Range: 0 to 1",
    )
    late_diffusion: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Sets how much diffusion is present in the latter part of the reverberation. Range: 0 to 1",
    )
    bandwidth: float = Field(
        default=8000,
        ge=0,
        le=24000,
        description="Sets the low pass cutoff frequency of the reverb input.",
    )


class ReverbRoom(_ReverbBaseModel[MonoPlacement]):
    """Mono Reverb room model."""

    op_type: Literal["ReverbRoom"] = "ReverbRoom"
    parameters: ReverbRoomParameters = Field(default_factory=ReverbRoomParameters)
    config: ReverbRoomConfig = Field(default_factory=ReverbRoomConfig)


class ReverbRoomStereo(_ReverbBaseModel[StereoPlacement]):
    """Stereo Reverb room model."""

    op_type: Literal["ReverbRoomStereo"] = "ReverbRoomStereo"
    parameters: ReverbRoomStereoParameters = Field(default_factory=ReverbRoomStereoParameters)
    config: ReverbRoomStereoConfig = Field(default_factory=ReverbRoomStereoConfig)


class ReverbPlateStereo(_ReverbBaseModel[StereoPlacement]):
    """
    The stereo room plate stage. This is based on Dattorro's 1997
    paper. This reverb consists of 4 allpass filters for input diffusion,
    followed by a figure of 8 reverb tank of allpasses, low-pass filters,
    and delays. The output is taken from multiple taps in the delay lines
    to get a desirable echo density.
    """

    op_type: Literal["ReverbPlateStereo"] = "ReverbPlateStereo"
    parameters: ReverbPlateStereoParameters = Field(default_factory=ReverbPlateStereoParameters)
    config: ReverbPlateStereoConfig = Field(default_factory=ReverbPlateStereoConfig)
