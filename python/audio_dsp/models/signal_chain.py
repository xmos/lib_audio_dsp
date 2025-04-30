from typing import Literal

from pydantic import Field, field_validator, model_validator

from audio_dsp.models.stage import NodePlacement, StageConfig, StageModel, StageParameters


class ForkConfig(StageConfig):
    count: int = Field(default=1)


class ForkPlacement(NodePlacement, extra="forbid"):
    input: list[int] = Field(default=set(), max_length=1, min_length=1)
    output: list[int] = Field(default=set())
    name: str
    thread: int = Field(ge=0, lt=5)

    @field_validator("input", "output", mode="before")
    def _single_to_list(cls, value: int | list) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]


class Fork(StageModel[ForkPlacement]):
    """Forks the input signal into multiple outputs."""

    op_type: Literal["Fork"] = "Fork"
    config: ForkConfig = Field(default_factory=ForkConfig)

    @model_validator(mode="after")
    def check_fork(self):
        try:
            in_len = len(self.placement.input)
        except TypeError:
            in_len = 1
        try:
            out_len = len(self.placement.output)
        except TypeError:
            out_len = 1

        if out_len / in_len != self.config.count:
            if out_len / in_len == out_len // in_len:
                self.config.count = out_len // in_len
            else:
                raise ValueError("number of fork outputs not a multiple of inputs")
        return self


class MixerParameters(StageParameters):
    gain_db: float = Field(default=0)


class MixerPlacement(NodePlacement, extra="forbid"):
    input: list[int] = Field(default=[])
    output: list[int] = Field(default=[], max_length=1, min_length=1)
    name: str
    thread: int = Field(ge=0, lt=5)

    @field_validator("input", "output", mode="before")
    def _single_to_list(cls, value: int | list) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]


class Mixer(StageModel[MixerPlacement]):
    """
    Mixes the input signals together. The mixer can be used to add signals together, or to attenuate the input signals.
    It must have exactly one output.
    """

    op_type: Literal["Mixer"] = "Mixer"
    parameters: MixerParameters = Field(default_factory=MixerParameters)


class Adder(StageModel):
    """
    Add the input signals together. The adder can be used to add signals
    together. It must have exactly one output.
    """

    op_type: Literal["Adder"] = "Adder"


class FixedGainParameters(StageParameters):
    gain_db: float = Field(default=0)


class FixedGain(StageModel):
    """
    This stage implements a fixed gain. The input signal is multiplied
    by a gain. If the gain is changed at runtime, pops and clicks may
    occur.

    If the gain needs to be changed at runtime, use a
    :class:`VolumeControl` stage instead.
    """

    # class Model(Stage.Model):
    op_type: Literal["FixedGain"] = "FixedGain"
    parameters: FixedGainParameters = Field(default_factory=FixedGainParameters)


class VolumeControlParameters(StageParameters):
    gain_db: float = Field(default=0)
    mute_state: int = Field(default=0)


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
    position: int = Field(default=0)


class Switch(StageModel):
    """
    Switch the input to one of the outputs. The switch can be used to
    select between different signals.

    """

    op_type: Literal["Switch"] = "Switch"
    parameters: SwitchParameters = Field(default_factory=SwitchParameters)


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
    
    Attributes:
        max_delay: Maximum delay length in samples
    """
    max_delay: int = Field(
        default=1024,
        gt=0,
        description="Maximum delay length in samples"
    )


class DelayParameters(StageParameters):
    """Parameters for delay stage.
    
    Attributes:
        delay: Current delay length in samples
        units: Units for delay values, either "samples" or "seconds"
    """
    delay: int = Field(
        default=0,
        ge=0,
        description="Current delay length in samples"
    )
    units: Literal["samples", "seconds"] = Field(
        default="samples",
        description="Units for delay values"
    )


class Delay(StageModel[NodePlacement]):
    """Delay stage for delaying input signals.
    
    Delays the input signal by a specified amount. The maximum delay is set at 
    compile time via config, and the runtime delay can be set between 0 and max_delay.
    The delay can be specified in either samples or seconds.
    """
    op_type: Literal["Delay"] = "Delay"
    parameters: DelayParameters = Field(default_factory=DelayParameters)
    config: DelayConfig = Field(default_factory=DelayConfig)
    placement: NodePlacement 