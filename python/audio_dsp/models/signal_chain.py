# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Pydantic models for signal chain DSP Stages."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from audio_dsp.models.stage import (
    NodePlacement,
    StageConfig,
    StageModel,
    StageParameters,
    Placement_2i1o,
    Placement_Ni1o,
    Placement_4i2o,
)
from audio_dsp.models.fields import DEFAULT_GAIN_DB
import annotated_types


class Bypass(StageModel):
    """
    This stage implements a bypass. The input signal is passed through
    unchanged.
    """

    op_type: Literal["Bypass"] = "Bypass"


class ForkConfig(StageConfig):
    """Compile time configuration for a Fork Stage."""

    count: int = Field(default=1)


class ForkPlacement(NodePlacement, extra="forbid"):
    """Graph placement for a Fork Stage."""

    input: list[int] = Field(default=[], description="List of input edges.", min_length=1)


class Fork(StageModel[ForkPlacement]):
    """Forks the input signal into multiple outputs."""

    op_type: Literal["Fork"] = "Fork"
    config: ForkConfig = Field(default_factory=ForkConfig)

    @model_validator(mode="after")
    def check_fork(self):
        """Check that the fork has been validly connected."""
        in_len = len(self.placement.input)
        out_len = len(self.placement.output)

        if out_len / in_len != self.config.count:
            if out_len / in_len == out_len // in_len:
                self.config.count = out_len // in_len
            else:
                raise ValueError("number of fork outputs not a multiple of inputs")
        return self


class MixerParameters(StageParameters):
    """Parameters for Mixer Stage."""

    gain_db: float = DEFAULT_GAIN_DB()


class Mixer(StageModel[Placement_Ni1o]):
    """
    Mixes the input signals together. The mixer can be used to add signals together, or to attenuate the input signals.
    It must have exactly one output.
    """

    op_type: Literal["Mixer"] = "Mixer"
    parameters: MixerParameters = Field(default_factory=MixerParameters)


class Adder(StageModel[Placement_Ni1o]):
    """
    Add the input signals together. The adder can be used to add signals
    together. It must have exactly one output.
    """

    op_type: Literal["Adder"] = "Adder"


class Subtractor(StageModel[Placement_2i1o]):
    """
    Subtract the input signals. The subtractor can be used to subtract
    signals together. It must have exactly one output.
    """

    op_type: Literal["Subtractor"] = "Subtractor"


class FixedGainParameters(StageParameters):
    """Parameters for FixedGain Stage."""

    gain_db: float = DEFAULT_GAIN_DB()


class FixedGain(StageModel):
    """
    This stage implements a fixed gain. The input signal is multiplied
    by a gain. If the gain is changed at runtime, pops and clicks may
    occur.

    If the gain needs to be changed at runtime, use a
    :class:`VolumeControl` stage instead.
    """

    op_type: Literal["FixedGain"] = "FixedGain"
    parameters: FixedGainParameters = Field(default_factory=FixedGainParameters)


class VolumeControlParameters(StageParameters):
    """Parameters for VolumeControl Stage."""

    gain_db: float = DEFAULT_GAIN_DB()
    mute_state: int = Field(default=0, ge=0, le=1, description="Mute state of the stage")


class VolumeControl(StageModel):
    """
    This stage implements a volume control. The input signal is
    multiplied by a gain. The gain can be changed at runtime. To avoid
    pops and clicks during gain changes, a slew is applied to the gain
    update. The stage can be muted and unmuted at runtime.
    """

    op_type: Literal["VolumeControl"] = "VolumeControl"
    parameters: VolumeControlParameters = Field(default_factory=VolumeControlParameters)


class SwitchParameters(StageParameters):
    """Parameters for Switch Stage."""

    position: int = Field(default=0, ge=0, description="Switch position")


class Switch(StageModel[Placement_Ni1o]):
    """
    Switch the input to one of the outputs. The switch can be used to
    select between different signals.

    """

    op_type: Literal["Switch"] = "Switch"
    parameters: SwitchParameters = Field(default_factory=SwitchParameters)

    @model_validator(mode="after")
    def set_max_outputs(self):
        """Set the maximum number os switch positions."""
        max_val = len(self.placement.input)
        type(self.parameters).model_fields["position"].metadata.append(
            annotated_types.Le(max_val - 1)
        )

        return self


class SwitchSlew(Switch):
    """
    Switch the input to one of the outputs with slew. The switch can be used to
    select between different signals.

    """

    op_type: Literal["SwitchSlew"] = "SwitchSlew"  # pyright: ignore


class SwitchStereo(StageModel):
    """
    Switch the input to one of the stereo pairs of outputs. The switch
    can be used to select between different stereo signal pairs. The
    inputs should be passed in pairs, e.g. ``[0_L, 0_R, 1_L, 1_R, ...]``.
    Setting the switch position will output the nth pair.

    """

    op_type: Literal["SwitchStereo"] = "SwitchStereo"
    parameters: SwitchParameters = Field(default_factory=SwitchParameters)


class DelayConfig(StageConfig):
    """Configuration for delay stage.

    Attributes
    ----------
        max_delay: Maximum delay length in samples
        units: Units for delay values, either "samples" or "seconds"
    """

    max_delay: int = Field(default=1024, gt=0, description="Maximum delay length in samples")
    units: Literal["samples", "s", "ms"] = Field(
        default="samples", description="Units for maximum delay values"
    )


class DelayParameters(StageParameters):
    """Parameters for delay stage.

    Attributes
    ----------
        delay: Current delay length in the configured units
    """

    delay: float = Field(default=0, ge=0, description="Current delay length")


class Delay(StageModel):
    """Delay stage for delaying input signals.

    Delays the input signal by a specified amount. The maximum delay is set at
    compile time via config, and the runtime delay can be set between 0 and max_delay.
    The delay can be specified in either samples or seconds.
    """

    op_type: Literal["Delay"] = "Delay"
    parameters: DelayParameters = Field(default_factory=DelayParameters)
    config: DelayConfig = Field(default_factory=DelayConfig)

    @model_validator(mode="after")
    def set_max_delay(self):
        """Set the maximum delay value based on the configuration."""
        max_val = self.config.max_delay
        type(self.parameters).model_fields["delay"].metadata.append(annotated_types.Le(max_val))

        return self


class CrossfaderParameters(StageParameters):
    """Parameters for crossfader stage."""

    mix: float = Field(default=0.5, le=1, ge=0, description="Set the mix of the crossfader")


class Crossfader(StageModel[Placement_2i1o]):
    """Crossfader stage model."""

    op_type: Literal["Crossfader"] = "Crossfader"
    parameters: CrossfaderParameters = Field(default_factory=CrossfaderParameters)


class CrossfaderStereo(StageModel[Placement_4i2o]):
    """Stereo Crossfader stage model."""

    op_type: Literal["CrossfaderStereo"] = "CrossfaderStereo"
    parameters: CrossfaderParameters = Field(default_factory=CrossfaderParameters)
