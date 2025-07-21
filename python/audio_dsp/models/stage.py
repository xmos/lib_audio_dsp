# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Generic pydantic models for DSP Stages."""

from typing import Type, Union, Optional, Any, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class edgeProducerBaseModel(BaseModel):
    """The pydantic model defining an edge producer (e.g. DSP Stage)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _GlobalStageModels:
    """Class to hold some globals."""

    stages = []


class StageConfig(BaseModel, extra="ignore"):
    """The pydantic model defining the compile-time configurable configuration of a DSP Stage."""

    pass


class StageParameters(BaseModel, extra="ignore"):
    """The pydantic model defining the runtime configurable cparameters of a DSP Stage."""

    pass


class NodePlacement(BaseModel, extra="forbid"):
    """The pydantic model that defines the placement of a DSP Stage in the graph.

    By default this expects inputs and outputs for each stage.
    This may be subclassed for custom placement behaviour.
    """

    name: str
    input: list[Tuple[str, int]] = Field(
        default=[],
        description="List of input edges.",
    )
    thread: int = Field(ge=0, lt=5)

    # @field_validator("input", "output", mode="before")
    # def _single_to_list(cls, value: Union[int, list]) -> list:
    #     if isinstance(value, list):
    #         return value
    #     else:
    #         return [value]


class MonoPlacement(NodePlacement):
    """The placement of a mono stage that must have 1 input and 1 output."""

    input: list[Tuple[str, int]] = Field(
        default=[],
        description="List of input edges.",
        min_length=1,
        max_length=1,
    )


class StereoPlacement(NodePlacement):
    """The placement of a stereo stage that must have 2 inputs and 2 outputs."""

    input: list[Tuple[str, int]] = Field(
        default=[],
        description="List of input edges.",
        min_length=2,
        max_length=2,
    )


class Placement_2i1o(NodePlacement):
    """The placement of a stage that must have 2 inputs and 1 outputs."""

    input: list[Tuple[str, int]] = Field(
        default=[],
        description="List of input edges.",
        min_length=2,
        max_length=2,
    )


class Placement_4i2o(NodePlacement):
    """The placement of a stage that must have 2 inputs and 1 outputs."""

    input: list[Tuple[str, int]] = Field(
        default=[],
        description="List of input edges.",
        min_length=4,
        max_length=4,
    )


class Placement_4i1o(NodePlacement):
    """The placement of a stage that must have 4 inputs and 1 output."""

    input: list[Tuple[str, int]] = Field(
        default=[],
        description="List of input edges.",
        min_length=4,
        max_length=4,
    )


class Placement_Ni1o(NodePlacement, extra="forbid"):
    """Graph placement for a Stage that takes many input and one output."""

    pass


class StageModel[Placement: NodePlacement](edgeProducerBaseModel):
    """A generic pydantic model of a DSP Stage.

    Stages should subclass this and define their op_type, parameters (optional),
    compile-time config (optional), and specific placement requirements (optional).
    """

    placement: Placement

    def __init_subclass__(cls) -> None:
        """Add all subclasses of StageModel to a global list for querying."""
        super().__init_subclass__()
        _GlobalStageModels.stages.append(cls)


def all_models() -> dict[str, Type[StageModel]]:
    """Get a dict containing all stages in scope."""
    return {
        s.__name__: s
        for s in _GlobalStageModels.stages
        if "op_type" in s.model_fields and not s.__name__.startswith("_")
    }
