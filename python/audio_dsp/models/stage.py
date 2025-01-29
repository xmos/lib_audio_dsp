from typing import Any, Type, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from audio_dsp.design import plot
from audio_dsp.dsp.generic import dsp_block


class edgeProducerBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _GlobalStageModels:
    """Class to hold some globals."""

    stages = []


class StageConfig(BaseModel, extra="forbid"):
    pass


class StageParameters(BaseModel, extra="forbid"):
    pass


class NodePlacement(BaseModel, extra="forbid"):
    input: list[int] = Field(default=[])
    output: list[int] = Field(default=[])
    name: str
    thread: int = Field(ge=0, lt=5)

    @field_validator("input", "output", mode="before")
    def _single_to_list(cls, value: Union[int, list]) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]


class StageModel(edgeProducerBaseModel):
    # op_type: is not defined as this Stage cannot be pipelined
    config: SkipJsonSchema[Any] = Field(default_factory=StageConfig)
    parameters: SkipJsonSchema[Any] = Field(default_factory=StageParameters)
    placement: NodePlacement

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
    return {
        s.__name__: s
        for s in _GlobalStageModels.stages
        if "op_type" in s.model_fields and not s.__name__.startswith("_")
    }
