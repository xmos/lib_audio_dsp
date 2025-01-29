
from .stage import StageModel, StageParameters, StageConfig
from typing import Literal
from pydantic import Field, root_validator

class ForkConfig(StageConfig):
    count: int = Field(default=1)


class Fork(StageModel):
    op_type: Literal["Fork"] = "Fork"
    config: ForkConfig = Field(default_factory=ForkConfig)
    # input: list[int] = Field(default=[], min_length=1, max_length=1)

    @root_validator(pre=True)
    def check_fork(cls, values):
        cnt = values.get("config")["count"]
        try:
            in_len =len(values.get("input"))
        except TypeError:
            in_len = 1

        try:
            out_len =len(values.get("output"))
        except TypeError:
            out_len = 1

        if out_len/in_len != cnt:
            if out_len/in_len == out_len//in_len:
                values["config"]["count"] = out_len//in_len
            else:
                raise ValueError("number of fork outputs not a multiple of inputs")
        return values

class MixerParameters(StageParameters):
    gain_db: float = Field(default=0)

class Mixer(StageModel):
    """
    Mixes the input signals together. The mixer can be used to add signals
    together, or to attenuate the input signals.
    """

    op_type: Literal["Mixer"] = "Mixer"
    parameters: MixerParameters = Field(default_factory=MixerParameters)


class Adder(StageModel):
    """
    Add the input signals together. The adder can be used to add signals
    together.
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

    Parameters
    ----------
    gain_db : float, optional
        The gain of the mixer in dB.
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

    Parameters
    ----------
    gain_db : float, optional
        The gain of the mixer in dB.
    mute_state : int, optional
        The mute state of the Volume Control: 0: unmuted, 1: muted.
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


