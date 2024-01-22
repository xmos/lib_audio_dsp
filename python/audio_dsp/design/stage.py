

from .graph import Edge, Node
import yaml
from pathlib import Path

def find_config(name):
    ret = Path(__file__).parents[3]/"stage_config"/f"{name}.yaml"
    if not ret.exists():
        raise ValueError(f"{ret} does not exist")
    return ret

class StageOutput(Edge):
    def __init__(self, fs=48000, frame_size=1):
        super().__init__()
        # index of the multiple outputs that the source node has
        self.source_index = None
        # which input index is this
        self.dest_index = None
        self.fs = fs
        self.frame_size = frame_size
        # TODO edges will probably need an associated type audio vs. data etc.
        # self.type = q23
        
    def __repr__(self) -> str:
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
    def __init__(self, config, inputs):
        super().__init__(self)
        self.i = [i for i in inputs]
        for i, input in enumerate(self.i):
            input.set_dest(self)
            input.dest_index = i
        if self.i:
            self.fs = self.i[0].fs
            self.frame_size = self.i[0].frame_size
        else:
            self.fs = None

        self.n_in = len(self.i)
        self._o = None
        self.yaml_dict = yaml.load(Path(config).read_text(), Loader=yaml.Loader)
        # module dict contains 1 entry with the name of the module as its key
        self.name = next(iter(self.yaml_dict["module"].keys()))
        self._control_fields = {name: ValueControlField() for name in self.yaml_dict["module"][self.name].keys()}

    @property
    def o(self):
        """stage output channels"""
        if self._o is None:
            raise RuntimeError("Stage must add outputs with create_outputs in its __init__ method")
        return self._o

    def create_outputs(self, n_out):
        self._o = []
        for i in range(n_out):
            output = StageOutput()
            output.source_index = i
            output.set_source(self)
            self._o.append(output)

    def __setitem__(self, key, value):
        if key not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")
        self._control_fields[key].value = value

    def __getitem__(self, key):
        if key not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")
        return self._control_fields[key].value

    def set_control_field_cb(self, field, getter, setter=None):
        """
        Register callbacks for getting and setting control fields, to be called by classes which implement stage

        Args:
            field: str name of the field
            getter: a function which returns the current value
            setter: A function which accepts 1 argument that will be used as the new value
        """
        if field not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")

        self._control_fields[field] = PropertyControlField(getter, setter)

    def get_config(self):
        ret = {}
        for command_name, cf in self._control_fields.items():
            if cf.value is not None:
                ret[f"{self.name}_{command_name}"] = cf.value
        return ret

    def process(self, channels):
        raise NotImplementedError()
