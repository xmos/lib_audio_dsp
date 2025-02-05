from .stage import StageParameters, StageConfig, StageModel
from typing import Literal
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema


class ReverbBaseParameters(StageParameters):
    predelay: float = Field(
        default=15, ge=0, le=30, description="Set the predelay in milliseconds."
    )
    width: float = Field(default=1.0, ge=0, le=1, description="Range: 0 to 1")
    pregain: float = Field(
        default=0.5,
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
        description="It is not advised to increase this value above the "
        "default 0.015, as it can result in saturation inside "
        "the reverb delay lines.",
    )


class ReverbBaseConfig(StageConfig):
    predelay: float = Field(default=30)


class _ReverbBaseModel(StageModel):
    """
    The base class for reverb stages, containing pre delays, and wet/dry
    mixes and pregain.
    """

    # op_type: is not defined as this Stage cannot be pipelined
    config: ReverbBaseConfig = Field(default_factory=ReverbBaseConfig)


class ReverbPlateParameters(ReverbBaseParameters):
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
    early_diffusion: float = Field(default=0.2, ge=0, le=1, description="Range: 0 to 1")
    late_diffusion: float = Field(default=0.6, ge=0, le=1, description="Range: 0 to 1")
    bandwidth: float = Field(default=8000, ge=0, le=24000, description="Range: 0 to 1")


class ReverbPlateStereo(_ReverbBaseModel):
    """
    The stereo room plate stage. This is based on Dattorro's 1997
    paper. This reverb consists of 4 allpass filters for input diffusion,
    followed by a figure of 8 reverb tank of allpasses, low-pass filters,
    and delays. The output is taken from multiple taps in the delay lines
    to get a desirable echo density.
    """

    input: list[int] = Field(default=[], min_length=2, max_length=2)
    output: list[int] = Field(default=[], max_length=2)
    op_type: Literal["ReverbPlateStereo"] = "ReverbPlateStereo"
    parameters: ReverbPlateParameters = Field(
        default_factory=ReverbPlateParameters
    )
