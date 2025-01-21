

from typing import Type

import numpy
# from .graph import Edge, Node
import yaml
from pathlib import Path
from audio_dsp.design import plot
from audio_dsp.dsp.generic import dsp_block
from typing import Optional
from types import NotImplementedType

from typing import TypeVar, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, validator
from typing import Union, Optional


class edgeProducerBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input: list[int] = Field(default=[])
    output: list[int] = Field(default=[])

    @field_validator("input", "output", mode="before")
    def _single_to_list(cls, value: Union[int, list]) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]


class _GlobalStageModels:
    """Class to hold some globals."""

    stages = []

class StageConfig(BaseModel, extra="forbid"):
    pass


class StageParameters(BaseModel, extra="forbid"):
    pass


class StageModel(edgeProducerBaseModel):
    # op_type: is not defined as this Stage cannot be pipelined
    config: Any = Field(default_factory=StageConfig)
    parameters: Any = Field(default_factory=StageParameters)
    input: list[int] = Field(default=[])
    output: list[int] = Field(default=[])
    name: str
    thread: int = Field(ge=0, lt=5)


    @field_validator("config")
    @classmethod
    def _validate_config(cls, val):
        if issubclass(type(val), StageConfig):
            return val
        raise ValueError("config must be a subclass of StageConfig")

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, val):
        if issubclass(type(val), StageParameters):
            return val
        raise ValueError("parameters must be a subclass of StageParameters")

    # stage doesn't actually have a model
    # model: Model
    def __init_subclass__(cls) -> None:
        """Add all subclasses of Stage to a global list for querying."""
        super().__init_subclass__()
        _GlobalStageModels.stages.append(cls)


def all_models() -> dict[str, Type[StageModel]]:
    """Get a dict containing all stages in scope."""
    return {s.__name__: s for s in _GlobalStageModels.stages if "op_type" in s.model_fields}