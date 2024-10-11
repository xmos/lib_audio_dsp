from pydantic import (
    BaseModel,
    create_model,
    ValidationError,
    Field,
    validate_call,
    ConfigDict,
    NonNegativeInt,
    field_validator,
    ValidationInfo,
)
from typing import Type, Literal, Any, Optional
from enum import Enum, auto


class IncompleteTypeError(Exception):
    """Raise when a type does not have a required callback"""


class IncompatibleEdge(Exception):
    """raise when a buffer or edge is passed an edge that it does not support."""


class _IrBaseModel(BaseModel):
    class Config:
        # forbid extra fields
        extra = "forbid"
        validate_assignment = True


class BufferType(Enum):
    ONE_TO_ONE = auto()
    BIG_TO_SMALL = auto()
    SMALL_TO_BIG = auto()


class BufferRatio(_IrBaseModel):
    input: int
    output: int

    def is_integral(self):
        return not self.input % self.output or not self.output % self.input


class Buffer(_IrBaseModel):
    ratio: BufferRatio = BufferRatio(input=1, output=1)

    def ratio_f(self):
        return self.ratio.output / self.ratio.input

    @property
    def type(self):
        if self.ratio.output > self.ratio.input:
            return BufferType.SMALL_TO_BIG
        elif self.ratio.output == self.ratio.input:
            return BufferType.ONE_TO_ONE
        else:
            return BufferType.BIG_TO_SMALL


class _StageBaseModel(_IrBaseModel):
    thread: int = 0
    const_stage: bool = False


class TuningBaseModel(_IrBaseModel):
    pass


class ConstantsBaseModel(_IrBaseModel):
    pass


def _default_determine_outputs_func(context, input_types):
    raise IncompleteTypeError(f"{context.stage.type} stage has no determine_outputs")


def _comma_sep(l: list[str]):
    return ", ".join(l)


def _default_c_process(context, state_var, input_vars, output_vars):
    return [
        f"void* inputs[] = {{{_comma_sep(input_vars)}}};",
        f"void* outputs[] = {{{_comma_sep(output_vars)}}};",
        f"{context.stage.type}_process(inputs, outputs, {state_var});",
    ]


def _default_c_control(context, state_var, control_var):
    return [f"{context.stage.type}_control({state_var}, {control_var})"]


class DspStage:
    def __init__(self):
        self.this_tuning = TuningBaseModel
        self.this_constants = ConstantsBaseModel
        self.this_determine_outputs = _default_determine_outputs_func
        self.this_c_process = _default_c_process
        # self.this_c_init = _default_c_init
        self.this_c_control = _default_c_control

    @validate_call
    def tuning(self, model: Type[TuningBaseModel]):
        self.this_tuning = model
        return model

    @validate_call
    def constants(self, model: Type[ConstantsBaseModel]):
        self.this_constants = model
        return model

    def determine_outputs(self, determine_outputs_func):
        self.this_determine_outputs = determine_outputs_func
        return determine_outputs_func

    def c_process(self, c_process_func):
        self.this_c_process = c_process_func
        return c_process_func


def _model_requires_fields(model):
    """Determine if all the fields in a pydantic model are optional, returns true if so."""
    try:
        model()
        return False
    except ValidationError:
        return True


class StageHandler:
    def __init__(self, name: str, stage: DspStage):
        self.stage = stage

        tuning_required = _model_requires_fields(stage.this_tuning)
        constants_required = _model_requires_fields(stage.this_constants)

        tuning_spec = (stage.this_tuning, Field() if tuning_required else stage.this_tuning())
        constants_spec = (
            stage.this_constants,
            Field() if constants_required else stage.this_constants(),
        )

        self.config_model = create_model(
            name,
            type=(Literal[name], Field(default=name)),
            tuning=tuning_spec,
            constants=constants_spec,
            __base__=_StageBaseModel,
        )

    def determine_outputs(self, context, input_types):
        return self.stage.this_determine_outputs(context, input_types)

    def c_process(self, context, state_var, input_vars, output_vars):
        return self.stage.this_c_process(context, state_var, input_vars, output_vars)


class EdgeConnection(_IrBaseModel):
    name: Optional[str] = None
    """Input and output edges don't need a name"""

    index: NonNegativeInt

    class Config:
        # make this class immutable, also triggers pydantic to add an
        # implementation of hash
        frozen = True


class UnknownEdge(_IrBaseModel):
    name: Literal["unknown"] = "unknown"


class EdgeFirstPass(_IrBaseModel):
    dsp_input: bool = False
    """pipeline input edge"""

    dsp_output: bool = False
    """pipeline output edge"""

    source: EdgeConnection
    destination: EdgeConnection

    type: UnknownEdge | dict[str, Any] = UnknownEdge()

    @field_validator("source")
    def _source_valid(cls, v, info: ValidationInfo):
        if v.name and info.data["dsp_input"]:
            raise ValueError("Input edges must not have a source name.")
        return v

    @field_validator("destination")
    def _destination_valid(cls, v, info: ValidationInfo):
        if v.name and info.data["dsp_output"]:
            raise ValueError("Output edges must not have a destination name.")
        return v

    def repr_route(self):
        return f"{self.source.name}[{self.source.index}] -> {self.destination.name}[{self.destination.index}]"


class EdgeShape(_IrBaseModel):
    pass


class EdgeConfig(_IrBaseModel):
    pass


def _default_buffer_transform(context, input_edges):
    raise IncompleteTypeError(f"{context.edge.type} has no buffer_transform")


def _default_c_definition(context, edge_var_name):
    raise IncompleteTypeError(f"{context.edge.type} has no c_definition")


def _default_c_buffer_write(context, edge_var_name):
    raise IncompleteTypeError(f"{context.edge.type} has no c_buffer_write")


def _default_c_buffer_read(context, edge_var_name):
    raise IncompleteTypeError(f"{context.edge.type} has no c_buffer_read")


def _default_c_sizeof(context, edge_var_name):
    raise IncompleteTypeError(f"{context.edge.type} has no c_sizeof")


class DspEdge:
    def __init__(self):
        self.this_shape = EdgeShape
        self.this_config = EdgeConfig
        self.this_buffer_transform = _default_buffer_transform
        self.this_c_definition = _default_c_definition
        self.this_c_buffer_write = _default_c_buffer_write
        self.this_c_buffer_read = _default_c_buffer_read
        self.this_c_sizeof = _default_c_sizeof

    @validate_call
    def shape(self, model: type[EdgeShape]):
        self.this_shape = model
        return model

    @validate_call
    def config(self, model: type[EdgeConfig]):
        try:
            model()
        except ValidationError as e:
            raise IncompleteTypeError("Edge config must have defaults for all fields.") from e
        self.this_config = model
        return model

    def buffer_transform(self, transform_func):
        """decorate a function which takes input edges and returns corresponding outputs where possible"""
        self.this_buffer_transform = transform_func
        return transform_func

    def c_definition(self, c_definition_func):
        self.this_c_definition = c_definition_func
        return c_definition_func

    def c_buffer_write(self, c_buffer_write_func):
        self.this_c_buffer_write = c_buffer_write_func
        return c_buffer_write_func

    def c_buffer_read(self, c_buffer_read_func):
        self.this_c_buffer_read = c_buffer_read_func
        return c_buffer_read_func

    def c_sizeof(self, c_sizeof_func):
        self.this_c_sizeof = c_sizeof_func
        return c_sizeof_func


class EdgeHandler:
    def __init__(self, name: str, edge: DspEdge):
        self.edge = edge
        shape_required = _model_requires_fields(edge.this_shape)
        config_required = _model_requires_fields(edge.this_config)

        shape_spec = (edge.this_shape, ...)
        config_spec = (edge.this_config, Field(default_factory=edge.this_config))
        self.config_model = create_model(
            name,
            name=(Literal[name], Field(default=name)),
            shape=shape_spec,
            config=config_spec,
        )

    def buffer_transform(self, context, input_edges):
        return self.edge.this_buffer_transform(context, input_edges)

    def c_definition(self, context, edge_var_name):
        return self.edge.this_c_definition(context, edge_var_name)

    def c_buffer_read(self, context, edge_var_name):
        return self.edge.this_c_buffer_read(context, edge_var_name)

    def c_buffer_write(self, context, edge_var_name):
        return self.edge.this_c_buffer_write(context, edge_var_name)

    def c_sizeof(self, context):
        return self.edge.this_c_sizeof(context)
