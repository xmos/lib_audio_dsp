"""Models for delay stages."""

from typing import Literal
from pydantic import Field

from audio_dsp.models.stage import StageModel, StageParameters


class DelayParameters(StageParameters):
    """Parameters for delay stage.
    
    Attributes:
        max_delay: Maximum delay length in samples
        starting_delay: Initial delay length in samples
        units: Units for delay values, either "samples" or "seconds"
    """
    max_delay: int = Field(
        default=1024,
        gt=0,
        description="Maximum delay length in samples"
    )
    starting_delay: int = Field(
        default=0,
        ge=0,
        description="Initial delay length in samples"
    )
    units: Literal["samples", "seconds"] = Field(
        default="samples",
        description="Units for delay values"
    )


class Delay(StageModel):
    """Delay stage for delaying input signals.
    
    Delays the input signal by a specified amount. The maximum delay is set at 
    compile time, and the runtime delay can be set between 0 and max_delay.
    The delay can be specified in either samples or seconds.
    """
    op_type: Literal["Delay"] = "Delay"
    parameters: DelayParameters = Field(default_factory=DelayParameters) 