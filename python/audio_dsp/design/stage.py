

from .graph import Edge, Node
import yaml


class StageOutput(Edge):
    def __init__(self):
        super().__init__()
        # index of the multiple outputs that the source node has
        self.source_index = None
        # which input index is this
        self.dest_index = None

class Stage(Node):
    def __init__(self, config, graph, inputs):
        super().__init__(self)
        self.i = [i for i in inputs]
        for i, input in enumerate(self.i):
            input.set_dest(self)
            input.dest_index = i

        self.n_in = len(self.i)
        self._o = None
        self._graph = graph
        self.yaml_dict = yaml.load(config, Loader=yaml.Loader)
        # module dict contains 1 entry with the name of the module as its key
        self.name = next(iter(self.yaml_dict["module"].keys()))
        self._control_fields = {name: None for name in self.yaml_dict["module"][self.name].keys()}

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
            self._graph.add_edge(output)
            self._o.append(output)

    def __setitem__(self, key, value):
        if key not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")
        self._control_fields[key] = value

    def __getitem__(self, key):
        if key not in self._control_fields:
            raise KeyError(f"{key} is not a valid control field for {self.name}, try one of {', '.join(self._control_fields.keys())}")
        return self._control_fields[key]

    def get_config(self):
        ret = {}
        for command_name, cf in self._control_fields.items():
            if cf is not None:
                ret[f"{self.name}_{command_name}"] = cf
        return ret

    def process(self, channels):
        raise NotImplementedError()
