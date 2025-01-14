# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The edges and nodes for a DSP pipeline."""

from typing import Type

import numpy
from .graph import Edge, Node
import yaml
from pathlib import Path
from audio_dsp.design import plot
from audio_dsp.dsp.generic import dsp_block
from typing import Optional
from types import NotImplementedType

from typing import TypeVar, Any
from pydantic import BaseModel
from pydantic import BaseModel, Field, field_validator, ConfigDict, validator
from typing import Union, Optional


def find_config(name):
    """
    Find the config yaml file for a stage by looking for it
    in the default directory for built in stages.

    Parameters
    ----------
    name : str
        Name of stage, e.g. a stage whose config is saved in "biquad.yaml"
        should pass in "biquad".

    Returns
    -------
    Path
        Path to the config file.
    """
    ret = Path(__file__).parents[3] / "stage_config" / f"{name}.yaml"
    if not ret.exists():
        raise ValueError(f"{ret} does not exist")
    return ret


class StageOutput(Edge):
    """
    The Edge of a dsp pipeline.

    Parameters
    ----------
    fs : int
        Edge sample rate Hz
    frame_size : int
        Number of samples per frame

    Attributes
    ----------
    source : audio_dsp.design.graph.Node
        Inherited from Edge
    dest : audio_dsp.design.graph.Node
        Inherited from Edge
    source_index : int | None
        The index of the edge connection to source.
    fs : int
        see fs parameter
    frame_size : int
        see frame_size parameter
    """

    def __init__(self, fs=48000, frame_size=1):
        super().__init__()
        # index of the multiple outputs that the source node has
        self.source_index = None
        # which input index is this
        self._dest_index = None
        self.fs = fs
        self.frame_size = frame_size
        # edges will probably need an associated type audio vs. data etc.
        # self.type = q23

    @property
    def dest_index(self) -> int | None:
        """The index of the edge connection to the dest."""
        return self._dest_index

    @dest_index.setter
    def dest_index(self, value):
        if self._dest_index is not None:
            raise RuntimeError(
                f"This edge has already been connected, edges cannot have multiple destinations."
            )
        self._dest_index = value

    def __repr__(self) -> str:
        """Make print output usable."""
        dest = "-" if self.dest is None else f"{self.dest.index} {self.dest_index}"
        source = "-" if self.source is None else f"{self.source.index} {self.source_index}"
        return f"({source} -> {dest})"


class StageOutputList:
    """
    A container of StageOutput.

    A stage output list will be created whenever a stage is added to the pipeline.
    It is unlikely that a StageOutputList will have to be explicitly created during pipeline
    design. However the indexing and combining methods shown in the example will be used to
    create new StageOutputList instances.

    Examples
    --------
    This example shows how to combine StageOutputList in various ways::

        # a and b are StageOutputList
        a = some_stage.o
        b = other_stage.o

        # concatenate them
        a + b

        # Choose a single channel from 'a'
        a[0]

        # Choose channels 0 and 3 from 'a'
        a[0, 3]

        # Choose a slice of channels from 'a', start:stop:step
        a[0:10:2]

        # Combine channels 0 and 3 from 'a', and 2 from 'b'
        a[0, 3] + b[2]

        # Join 'a' and 'b', with a placeholder "None" in between
        a + None + b

    Attributes
    ----------
    edges : list[StageOutput]
        To access the actual edges contained within this list then read from the edges
        attribute. All methods in this class return new StageOutputList instances (even
        when the length is 1).

    Parameters
    ----------
    edges
        list of StageOutput to create this list from.
    """

    def __init__(self, edges: list[StageOutput | None] | None = None):
        edges = edges or []
        for edge in edges:
            if not isinstance(edge, StageOutput) and edge is not None:
                raise TypeError(
                    f"Expected iterable of StageOutput or None, however it contained a {type(edge)}"
                )
        self.edges = edges

    def __iter__(self):
        """Iterate through the edges, yielding a new StageOutputList for each edge."""
        for e in self.edges:
            yield StageOutputList([e])

    def __len__(self):
        """Get the number of edges in this list."""
        return len(self.edges)

    def __radd__(self, other) -> "StageOutputList | NotImplementedType":
        """Other + self."""
        if isinstance(other, list):
            other = StageOutputList(other)
        if other is None:
            other = StageOutputList([None])
        if other == 0:
            # special case for sum(stage_outputs)
            return StageOutputList(self.edges)
        if not isinstance(other, StageOutputList):
            return NotImplemented
        return StageOutputList(other.edges + self.edges)

    def __add__(self, other) -> "StageOutputList | NotImplementedType":
        """Create a new StageOutputList which concatenates the input lists."""
        if isinstance(other, list):
            other = StageOutputList(other)
        if other is None:
            other = StageOutputList([None])
        if not isinstance(other, StageOutputList):
            return NotImplemented
        return StageOutputList(self.edges + other.edges)

    def __or__(self, other) -> "StageOutputList":
        """Support for '|', does the same as __add__."""
        return self + other

    def __ror__(self, other) -> "StageOutputList":
        """Support for '|', does the same as __radd__."""
        return other + self

    def __getitem__(self, key) -> "StageOutputList":
        """Create new StageOutputList containing requested indices from this one."""
        if isinstance(key, slice):
            return StageOutputList(self.edges[key])
        elif isinstance(key, int):
            return StageOutputList([self.edges[key]])
        else:
            return StageOutputList([self.edges[i] for i in key])

    def __eq__(self, other):
        """Check if this list contains the same edges as another."""
        if other is None:
            return False
        else:
            return all(a is b for a, b in zip(self.edges, other.edges))


class PropertyControlField:
    """For stages which have internal state they can register callbacks
    for getting and setting control fields.
    """

    def __init__(self, get, set=None):
        self._getter = get
        self._setter = set

    @property
    def value(self):
        """
        The current value of this control field.

        Determined by executing the getter method.
        """
        return self._getter()

    @value.setter
    def value(self, value):
        if self._setter is None:
            raise RuntimeError("This control field can't be set directly")
        self._setter(value)


class ValueControlField:
    """Simple field which can be updated directly."""

    def __init__(self, value=None):
        self.value = value


class _GlobalStages:
    """Class to hold some globals."""

    stages = []


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


class StageConfig(BaseModel, extra="forbid"):
    pass


class StageParameters(BaseModel, extra="forbid"):
    pass


# This defines the types of instances of the config/parameter classes
StageParameterType = TypeVar("StageParameterType", bound="StageParameters")
# StageConfigType = TypeVar("StageConfigType", bound="StageConfig")


class Stage(Node):
    """
    Base class for stages in the DSP pipeline. Each subclass
    should have a corresponding C implementation. Enables
    code generation, tuning and simulation of a stage.

    The stages config can be written and read using square brackets as with a
    dictionary. This is shown in the below example, note that the config field
    must have been declared in the stages yaml file.

        self["config_field"] = 2
        assert self["config_field"] == 2

    Parameters
    ----------
    config : str | Path
        Path to yaml file containing the stage definition for this stage. Config
        parameters are derived from this config file.
    inputs : Iterable[StageOutput]
        Pipeline edges to connect to self
    name : str
        Name of the stage. Passed instead of config when the stage does not have
        an associated config yaml file
    label : str
        User defined label for the stage. Used for autogenerating a define for accessing the stage's index
        in the device code

    Attributes
    ----------
    i : list[StageOutput]
        This stages inputs.
    fs : int | None
        Sample rate.
    frame_size : int | None
        Samples in frame.
    name : str
        Stage name determined from config file
    yaml_dict : dict
        config parsed from the config file
    label : str
        User specified label for the stage
    n_in : int
        number of inputs
    n_out : int
        number of outputs
    details : dict
        Dictionary of descriptive details which can be displayed to describe
        current tuning of this stage
    dsp_block : None | audio_dsp.dsp.generic.dsp_block
        This will point to a dsp block class (e.g. biquad etc), to be set by the child class
    """

    def __init__(
        self,
        inputs: StageOutputList,
        config: Optional[Path | str] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
    ):
        super().__init__()
        self.i = inputs[:]
        for i, input in enumerate(self.i.edges):
            if input is None:
                raise TypeError("All stage inputs must not be None")
            input.set_dest(self)
            input.dest_index = i
        if self.i:
            assert self.i.edges[0] is not None, "not possible as checked above"
            self.fs = self.i.edges[0].fs
            self.frame_size = self.i.edges[0].frame_size
        else:
            self.fs = None
            self.frame_size = None

        self.n_in = len(self.i)
        self.n_out = 0
        self._o = None
        if (config is None and name is None) or (config is not None and name is not None):
            raise RuntimeError("Provide either config or name, not both or none.")
        if config is not None:
            self.yaml_dict = yaml.load(Path(config).read_text(), Loader=yaml.Loader)
            # module dict contains 1 entry with the name of the module as its key
            self.name = next(iter(self.yaml_dict["module"].keys()))
            self._control_fields = {
                name: ValueControlField() for name in self.yaml_dict["module"][self.name].keys()
            }
        elif name is not None:
            self.name = name
            self._control_fields = {}
            self.yaml_dict = None

        self._constants = {}

        self.label = label

        self.details = {}
        self.dsp_block: Optional[dsp_block] = None
        self.stage_memory_string: str = ""
        self.stage_memory_parameters: tuple | None = None

    def __init_subclass__(cls) -> None:
        """Add all subclasses of Stage to a global list for querying."""
        super().__init_subclass__()
        _GlobalStages.stages.append(cls)

    class Model(edgeProducerBaseModel):
        # op_type: is not defined as this Stage cannot be pipelined
        config: Any = Field(default_factory=StageConfig)
        parameters: Any = Field(default_factory=StageParameters)
        input: list[int] = Field(default=[])
        output: list[int] = Field(default=[])
        name: str
        thread: int

        @validator("config")
        def _validate_config(cls, val):
            if issubclass(type(val), StageConfig):
                return val
            raise TypeError("config must be a subclass of StageConfig")

        @validator("parameters")
        def _validate_parameters(cls, val):
            if issubclass(type(val), StageParameters):
                return val
            raise TypeError("parameters must be a subclass of StageParameters")

    # stage doesn't actually have a model
    # model: Model

    def set_parameters(self, parameters: StageParameterType):
        pass

    @property
    def o(self) -> StageOutputList:
        """
        This stage's outputs. Use this object to connect this stage to the next stage in the pipeline.
        Subclass must call self.create_outputs() for this to exist.
        """
        if self._o is None:
            raise RuntimeError("Stage must add outputs with create_outputs in its __init__ method")
        return self._o

    def create_outputs(self, n_out):
        """
        Create this stages outputs.

        Parameters
        ----------
        n_out : int
            number of outputs to create.
        """
        self.n_out = n_out
        o = []
        for i in range(n_out):
            output = StageOutput(fs=self.fs, frame_size=self.frame_size)
            output.source_index = i
            output.set_source(self)
            o.append(output)
        self._o = StageOutputList(o)

    def __setitem__(self, key, value):
        """Support for dictionary like access to config fields."""
        if key not in self._control_fields:
            raise KeyError(
                f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}"
            )
        self._control_fields[key].value = value

    def __getitem__(self, key):
        """Support for dictionary like access to config fields."""
        if key not in self._control_fields:
            raise KeyError(
                f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}"
            )
        return self._control_fields[key].value

    def set_control_field_cb(self, field, getter, setter=None):
        """
        Register callbacks for getting and setting control fields, to be called by classes which implement stage.

        Parameters
        ----------
        field : str
            name of the field
        getter : function
            A function which returns the current value
        setter : function
            A function which accepts 1 argument that will be used as the new value
        """
        if field not in self._control_fields:
            raise KeyError(
                f"{field} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}"
            )

        self._control_fields[field] = PropertyControlField(getter, setter)

    def set_constant(self, field, value, value_type):
        """
        Define constant values in the stage. These will be hard coded in
        the autogenerated code and cannot be changed at runtime.

        Parameters
        ----------
        field : str
            name of the field
        value : ndarray or int or float or list
            value of the constant. This can be an array or scalar

        """
        if not isinstance(value, (int, float, numpy.ndarray, list)):
            raise TypeError(f"Type {type(value)} not a supported Stage constant value format")

        if isinstance(value, numpy.ndarray) and value.ndim > 1:
            raise TypeError(f"Only 1D numpy arrays can be set as Stage constants")

        self._constants[field] = value

    @property
    def constants(self):
        """Get a copy of the constants for this stage."""
        # Copy so that the caller cannot modify
        return {k: v for k, v in self._constants.items()}

    def get_config(self):
        """Get a dictionary containing the current value of the control
        fields which have been set.

        Returns
        -------
        dict
            current control fields
        """
        ret = {}
        for command_name, cf in self._control_fields.items():
            if cf.value is not None:
                ret[command_name] = cf.value
        return ret

    def process(self, in_channels):
        """
        Run dsp object on the input channels and return the output.

        Args:
            in_channels: list of numpy arrays

        Returns
        -------
            list of numpy arrays.
        """
        # use float implementation as it is faster
        return self.dsp_block.process_frame(in_channels)

    def get_frequency_response(self, nfft=512) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Return the frequency response of this instance's dsp_block attribute.

        Parameters
        ----------
        nfft
            The length of the FFT

        Returns
        -------
        ndarray, ndarray
            Frequency values, Frequency response for this stage.
        """
        if self.dsp_block is None:
            raise RuntimeError("This stage has not set its dsp_block")
        return self.dsp_block.freq_response(nfft)

    def plot_frequency_response(self, nfft=512):
        """
        Plot magnitude and phase response of this stage using matplotlib. Will
        be displayed inline in a jupyter notebook.

        Parameters
        ----------
        nfft : int
            Number of frequency bins to calculate in the fft.
        """
        f, h = self.get_frequency_response(nfft)
        plot.plot_frequency_response(f, h, name=self.name)

    def add_to_dot(self, dot):
        """
        Add this stage to a diagram that is being constructed.
        Does not add the edges.

        Parameters
        ----------
        dot : graphviz.Diagraph
            dot instance to add edges to.
        """
        inputs = "|".join(f"<i{i}> " for i in range(self.n_in))
        outputs = "|".join(f"<o{i}> " for i in range(self.n_out))
        center = f"{self.index}: {type(self).__name__}\\n"

        if self.label:
            center = f"{self.index}: {self.label}\\n"
        if self.details:
            details = "\\n".join(f"{k}: {v}" for k, v in self.details.items())
            label = f"{{ {{ {inputs} }} | {center} | {details} | {{ {outputs} }}}}"
        else:
            label = f"{{ {{ {inputs} }} | {center} | {{ {outputs} }}}}"

        dot.node(self.id.hex, label)

    def get_required_allocator_size(self):
        """
        Calculate the required statically-allocated memory in bytes for this stage.
        Formats this into a compile-time determinable expression.

        Returns
        -------
            compile-time determinable expression of required allocator size.
        """
        macro_name = f"{self.name.upper()}_STAGE_REQUIRED_MEMORY"
        if self.stage_memory_parameters is not None:
            return f"{macro_name}({','.join((str(x) for x in self.stage_memory_parameters))})"
        else:
            return macro_name


def all_stages() -> dict[str, Type[Stage]]:
    """Get a dict containing all stages in scope."""
    return {s.__name__: s for s in _GlobalStages.stages}


def all_useable_stages() -> dict[str, Type[Stage]]:
    """Get a dict containing all stages useable via the JSON interface."""
    return {s.__name__: s for s in _GlobalStages.stages if "op_type" in s.Model.model_fields}
