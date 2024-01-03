from typing import Iterable

from .graph import Edge, Graph, Node
from .stage import StageOutput


class Thread:
    def __init__(self, id: int, graph: Graph):
        self.id = id

        # self.i = [i for i in inputs]  # just the inputs
        # self.o = []
        self._graph = graph
        self._nodes = []

    def stage(self, stage_type, inputs: Iterable[StageOutput], **kwargs):
        """
        creates an instance of stage_type and registers it with the graph and also keeps a reference to it 

        returns the created stage
        """
        stage = stage_type(graph=self._graph, inputs=inputs, **kwargs)
        self._graph.add_node(stage)
        self._nodes.append(stage)
        return stage

    def stages(self, stage_types, inputs: Iterable[StageOutput]):
        """
        Same as stage but takes an array of stage type and connects them together

        Returns a list of the created instances.
        """
        ret = []
        for stage_type in stage_types:
            s = self.stage(stage_type, inputs)
            ret.append(s)
            inputs = s.o
        return ret

    # def output(self, outputs: Iterable[StageOutput]):
    #     """Set the outputs of this thread"""
    #     for o in outputs:
    #         self.o.append(o)

    def contains_stage(self, stage) -> bool:
        return stage in self._nodes

    def __enter__(self):
        return self

    def __exit__(self ,type, value, traceback):
        ...

        
