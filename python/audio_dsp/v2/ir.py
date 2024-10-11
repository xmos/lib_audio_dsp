import yaml
from typing import Literal, Any, Optional, Annotated, Union
from pathlib import Path
from pydantic import BaseModel, NonNegativeInt, Field, field_validator
from copy import deepcopy
import importlib
from .types import StageHandler, EdgeFirstPass, EdgeHandler, UnknownEdge, Buffer


THIS_DIR = Path(__file__).parent
DEFAULT_INCLUDES = [THIS_DIR / "builtin_stages.yaml"]

# TODO should all the models be in types.py?


class _IrBaseModel(BaseModel):
    class Config:
        # forbid extra fields
        extra = "forbid"
        validate_assignment = True  # TODO very slow


class EdgeDefinition(_IrBaseModel):
    type: Literal["intrinsic"] = "intrinsic"
    location: str

    def load(self, name) -> EdgeHandler:
        module_str, attr = self.location.split(":")
        module = importlib.import_module(module_str)
        return EdgeHandler(name, getattr(module, attr))


class StageDefinition(_IrBaseModel):
    type: Literal["intrinsic"] = "intrinsic"
    location: str

    def load(self, name) -> StageHandler:
        module_str, attr = self.location.split(":")
        module = importlib.import_module(module_str)
        return StageHandler(name, getattr(module, attr))


class Settings(_IrBaseModel):
    name: str


class ConfigTopLevelBase(_IrBaseModel):
    settings: Settings
    buffers: dict[str, Buffer]
    stage_definitions: dict[str, StageDefinition] = {}
    edge_definitions: dict[str, EdgeDefinition] = {}

    def load_stage(self, name):
        return self.stage_definitions[name].load(name)

    def load_edge(self, name):
        return self.edge_definitions[name].load(name)


class ConfigTopLevelFirstPass(ConfigTopLevelBase):
    stages: dict[str, Any]
    edges: list[EdgeFirstPass]


class IncludeConfig(_IrBaseModel):
    stage_definitions: dict[str, StageDefinition] = {}
    edge_definitions: dict[str, EdgeDefinition] = {}


def _load_intrinsic_stage(config: StageDefinition):
    module_str, attr = StageDefinition.location.split(":")
    module = importlib.import_module(module_str)
    return getattr(module, attr)


def _read_yaml_docs(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _stage_model(stage_handlers: dict[str, StageHandler]):
    """
    Create a pydantic type which is a tagged union of the config of all
    the declared stages. Discriminated by the "type" field.
    """
    union_type = Union[tuple(i.config_model for i in stage_handlers.values())]
    return Annotated[union_type, Field(discriminator="type")]


def _edge_model(edge_handlers: dict[str, EdgeHandler]):
    """
    Create a pydantic type which is a tagged union of the config of all
    the declared edges. Discriminated by the "type" field.
    """
    union_type = Union[(UnknownEdge, *(i.config_model for i in edge_handlers.values()))]
    return Annotated[union_type, Field(discriminator="name")]


def _dict_merge_unique(orig, new):
    """Update a dictionary, raising a valueerror if the key was already present"""
    ret = {**orig}
    for k, v in new.items():
        if k in ret:
            raise ValueError(f"{k} exists in both")
        ret[k] = v
    return ret


class EdgeContext(BaseModel):
    stage_definitions: dict[str, StageDefinition] = {}
    edge_definitions: dict[str, EdgeDefinition] = {}
    edge: Any


class StageContext(BaseModel):
    stage_definitions: dict[str, StageDefinition] = {}
    edge_definitions: dict[str, EdgeDefinition] = {}
    stage: Any


class IR:
    """Holder for the ir with helper methods"""

    def __init__(self, config_struct: ConfigTopLevelBase, edge_model):
        self.config_struct = config_struct
        self.edge_model = edge_model

    def get_node_inputs(self, node: str):
        """Filter the edges by the dest address"""
        edges = [
            (i, e)
            for i, e in enumerate(self.config_struct.edges)
            if e.destination and e.destination.name == node
        ]
        edges = sorted(edges, key=lambda x: x[1].destination.index)
        return dict(edges)

    def get_node_outputs(self, node: str):
        """Filter edges by the source address"""
        edges = [
            (i, e)
            for i, e in enumerate(self.config_struct.edges)
            if e.source and e.source.name == node
        ]
        edges = sorted(edges, key=lambda x: x[1].source.index)
        return dict(edges)

    def edge_context(self, i: int):
        """Context passed to edge type callback functions"""
        return EdgeContext(
            stage_definitions=self.config_struct.stage_definitions,
            edge_definitions=self.config_struct.edge_definitions,
            edge=self.config_struct.edges[i],
        )

    def edge_context_from_edge(self, edge):
        """Context passed to edge type callback functions"""
        return EdgeContext(
            stage_definitions=self.config_struct.stage_definitions,
            edge_definitions=self.config_struct.edge_definitions,
            edge=edge.model_copy(),
        )

    def stage_context(self, name: str):
        """Context passed to stage type callback functions"""
        return StageContext(
            stage_definitions=self.config_struct.stage_definitions,
            edge_definitions=self.config_struct.edge_definitions,
            stage=self.config_struct.stages[name],
        )

    def thread_ids(self):
        """Get a set containing all of the thread IDs in the IR"""
        ret = set()
        for stage in self.config_struct.stages.values():
            ret.add(stage.thread)
        return ret

    def input_indices(self):
        # set to remove dupes
        return sorted(list(set(i.source.index for i in self.config_struct.edges if i.dsp_input)))

    def output_indices(self):
        return sorted(list(i.destination.index for i in self.config_struct.edges if i.dsp_output))

    def split_stages_by_thread(self):
        """
        return a dictionary mapping thread index to a dictionary of the stages on that thread

        Returns a copy of each stage so modifications do not impact the original IR.
        """
        ret = {}
        for name, stage in self.config_struct.stages.items():
            try:
                ret[stage.thread][name] = stage.model_copy()
            except KeyError:
                ret[stage.thread] = {name: stage.model_copy()}
        return ret

    def copy(self) -> "IR":
        """Get a new IR with a copy of the config struct."""
        return IR(self.config_struct.model_copy(), self.edge_model)


def load_from_yaml(path):
    """
    Load the IR from a yaml file. Validate that each field in the yaml is valid. But
    do not validate the graph itself, that is done in the validation stage. All included
    yaml files are merged into a single ir object.
    """
    parsed = _read_yaml_docs(path)
    validate1 = ConfigTopLevelFirstPass(**parsed)
    stage_definitions = validate1.stage_definitions
    edge_definitions = validate1.edge_definitions

    # TODO let user specify extra includes
    include_yaml = [*DEFAULT_INCLUDES]
    for file in include_yaml:
        include_config = IncludeConfig(**_read_yaml_docs(file))
        stage_definitions = _dict_merge_unique(stage_definitions, include_config.stage_definitions)
        edge_definitions = _dict_merge_unique(edge_definitions, include_config.edge_definitions)

    # load all the stages
    stages = {}
    for name, definition in stage_definitions.items():
        stages[name] = definition.load(name)
    # load all the edges
    edges = {}
    for name, definition in edge_definitions.items():
        edges[name] = definition.load(name)

    # validate all the stages in the graph by creating a custom pydantic model that
    # can check the configuration for all stages and edges.
    stage_model = _stage_model(stages)
    edge_model = _edge_model(edges)

    class EdgeModelFinal(EdgeFirstPass):
        type: UnknownEdge | edge_model

    class ConfigTopLevel(ConfigTopLevelBase):
        stages: dict[str, stage_model]
        edges: list[EdgeModelFinal]

    validate2 = ConfigTopLevel(**validate1.model_dump())
    validate2.stage_definitions = stage_definitions
    validate2.edge_definitions = edge_definitions
    return IR(validate2, EdgeModelFinal)
