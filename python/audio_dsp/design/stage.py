
from .graph import Edge, Node
import yaml
from pathlib import Path


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
    ret = Path(__file__).parents[3]/"stage_config"/f"{name}.yaml"
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
    dest_index : int | None
        The index of the edge connection to dest.
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
        # TODO edges will probably need an associated type audio vs. data etc.
        # self.type = q23

    @property
    def dest_index(self):
        return self._dest_index

    @dest_index.setter
    def dest_index(self, value):
        if self._dest_index is not None:
            raise RuntimeError(f"This edge alread has a dest index, can't be changes to {value}")
        self._dest_index = value

    def __repr__(self) -> str:
        """
        Makes print output usable
        """
        dest = "-" if self.dest is None else f"{self.dest.index} {self.dest_index}"
        return f"({self.source.index} {self.source_index} -> {dest})"


class PropertyControlField:
    """For stages which have internal state they can register callbacks
    for getting and setting control fields"""
    def __init__(self, get, set=None):
        self._getter = get
        self._setter = set

    @property
    def value(self):
        return self._getter()

    @value.setter
    def value(self, value):
        if self._setter is None:
            raise RuntimeError("This control field can't be set directly")
        self._setter(value)


class ValueControlField:
    """Simple field which can be updated directly"""
    def __init__(self, value=None):
        self.value = value

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

    Attributes
    ----------
    i : list[StageOutput]
        This stages inputs.
    o : list[StageOutput]
        This stages outputs, use to connect to the next stage in the pipeline.
        Subclass must call self.create_outputs() for this to exist.
    fs : int | None
        Sample rate.
    frame_size : int | None
        Samples in frame.
    name : str
        Module name determined from config file
    yaml_dict : dict
        config parsed from the config file
    n_in : int
        number of inputs
    n_out : int
        number of outputs
    details : dict
        Dictionary of descriptive details which can be displayed to describe
        current tuning of this stage
    """
    def __init__(self, config, inputs):
        super().__init__()
        self.i = [i for i in inputs]
        for i, input in enumerate(self.i):
            input.set_dest(self)
            input.dest_index = i
        if self.i:
            self.fs = self.i[0].fs
            self.frame_size = self.i[0].frame_size
        else:
            self.fs = None
            self.frame_size = None

        self.n_in = len(self.i)
        self.n_out = 0
        self._o = None
        self.yaml_dict = yaml.load(Path(config).read_text(), Loader=yaml.Loader)
        # module dict contains 1 entry with the name of the module as its key
        self.name = next(iter(self.yaml_dict["module"].keys()))
        self._control_fields = {name: ValueControlField() for name in self.yaml_dict["module"][self.name].keys()}
        self.details = {}

    @property
    def o(self):
        if self._o is None:
            raise RuntimeError("Stage must add outputs with create_outputs in its __init__ method")
        return self._o

    def create_outputs(self, n_out):
        """
        Create this stages outputs.

        TODO fs and frame_size

        Parameters
        ----------
        n_out : int
            number of outputs to create.
        """
        self.n_out = n_out
        self._o = []
        for i in range(n_out):
            output = StageOutput(fs=self.fs, frame_size=self.frame_size)
            output.source_index = i
            output.set_source(self)
            self._o.append(output)

    def __setitem__(self, key, value):
        """Support for dictionary like access to config fields"""
        if key not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")
        self._control_fields[key].value = value

    def __getitem__(self, key):
        """Support for dictionary like access to config fields"""
        if key not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")
        return self._control_fields[key].value

    def set_control_field_cb(self, field, getter, setter=None):
        """
        Register callbacks for getting and setting control fields, to be called by classes which implement stage

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
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")

        self._control_fields[field] = PropertyControlField(getter, setter)

    def get_config(self):
        """Get a dictionary containing the current value of the control
        fields which have been set

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

    def process(self, channels):
        """
        TODO
        """
        raise NotImplementedError()

    def get_frequency_response(self, nfft=512):
        raise NotImplementedError()

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
        if self.details:
            details = "\\n".join(f"{k}: {v}" for k, v in self.details.items())
            label = f"{{ {{ {inputs} }} | {center} | {details} | {{ {outputs} }}}}"
        else:
            label = f"{{ {{ {inputs} }} | {center} | {{ {outputs} }}}}"
        dot.node(self.id.hex, label)
