# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Contains the higher order stage class CompositeStage."""

from typing import Iterable, Type, TypeVar

from .stage import StageOutput, Stage, StageOutputList
from .graph import Graph

from ._draw import new_record_digraph
from IPython import display
from uuid import uuid4

import itertools

_StageOrComposite = TypeVar("_StageOrComposite", bound="Stage | CompositeStage")


class CompositeStage:
    """
    This is a higher order stage.

    Contains stages as well as other composite
    stages. A thread will be a composite stage. Composite stages allow:

    - drawing the detail with graphviz
    - process
    - frequency response

    TODO:
    - Process method on the composite stage will need to know its inputs and
    the order of the inputs (which input index corresponds to each input edge).
    However a CompositeStage doesn't know all of its inputs when it is created.

    Parameters
    ----------
    graph : audio_dsp.graph.Graph
        instance of graph that all stages in this composite will be added to.
    name : str
        Name of this instance to use when drawing the pipeline, defaults to class name.
    """

    def __init__(self, graph: Graph, name: str = ""):
        self._graph = graph
        self._stages = []
        self._composite_stages = []
        self._name = name
        if not self._name:
            self._name = type(self).__name__

    def composite_stage(self, name: str = "") -> "CompositeStage":
        """
        Create a new composite stage that will be a
        included in the current composite. The
        new stage can have stages added to it
        dynamically.
        """
        new = CompositeStage(graph=self._graph, name=name)
        self._composite_stages.append(new)
        return new

    @property
    def o(self) -> StageOutputList:
        """
        Outputs of this composite.

        Dynamically computed by searching the graph for edges which
        originate in this composite and whose destination is outside this composite. Order
        not currently specified.
        """
        all_stages = self.get_all_stages()
        all_edges = list(itertools.chain.from_iterable([stage.o.edges for stage in all_stages]))
        return StageOutputList(
            [edge for edge in all_edges if edge is not None and edge.dest not in all_stages]
        )

    def stage(
        self,
        stage_type: Type[_StageOrComposite],
        inputs: StageOutputList,
        **kwargs,
    ) -> _StageOrComposite:
        """
        Create a new stage or composite stage and
        register it with this composite stage.

        Parameters
        ----------
        stage_type
            Must be a subclass of Stage or CompositeStage
        inputs
            Edges of the pipeline that will be connected to the newly created stage.
        kwargs : dict
            Additional args are forwarded to the stages constructors (__init__)

        Returns
        -------
        stage_type
            Newly created stage or composite stage.
        """
        if issubclass(stage_type, CompositeStage):
            # Subclasses of CompositeStage must have extra __init__
            # parameters that pyright cant know.
            stage = stage_type(inputs=inputs, graph=self._graph, **kwargs)  # type: ignore
            self._composite_stages.append(stage)
        elif issubclass(stage_type, Stage):
            stage = stage_type(inputs=inputs, **kwargs)
            self._graph.add_node(stage)
            for edge in stage.o.edges:
                self._graph.add_edge(edge)
            self._stages.append(stage)
        else:
            raise ValueError(f"{stage_type} is not a Stage or CompositeStage")
        return stage

    def stages(
        self, stage_types: list[Type[_StageOrComposite]], inputs: StageOutputList
    ) -> list[_StageOrComposite]:
        """
        Iterate through the provided stages and connect them linearly.

        Returns a list of the created instances.
        """
        ret = []
        for stage_type in stage_types:
            s = self.stage(stage_type, inputs)
            ret.append(s)
            inputs = s.o
        return ret

    def contains_stage(self, stage: Stage) -> bool:
        """
        Recursively search self for the stage.

        Returns
        -------
        bool
            True if this composite contains the stage else False
        """
        return stage in self.get_all_stages()

    def get_all_stages(self) -> list[Stage]:
        """
        Get a flat list of all stages contained within this composite stage
        and the composite stages within.

        Returns
        -------
            list of stages.
        """
        return sum([c.get_all_stages() for c in self._composite_stages], start=self._stages)

    def process(self, data):
        """
        Execute the stages in this composite on the host.

        .. warning::
            Not implemented.
        """
        raise NotImplementedError()

    def _internal_edges(self) -> list[StageOutput]:
        """Return a list of edges whose source and dest are within this composite."""
        all_stages = self.get_all_stages()
        all_edges = list(itertools.chain.from_iterable([stage.o.edges for stage in all_stages]))
        return [
            edge
            for edge in all_edges
            if edge is not None and edge.dest in all_stages and edge.source in all_stages
        ]

    def draw(self):
        """Draws the stages and edges present in this instance of a composite stage."""
        dot = new_record_digraph()
        self.add_to_dot(dot)
        output_edges = self.o
        internal_edges = self._internal_edges()
        for e in internal_edges:
            source = f"{e.source.id.hex}:o{e.source_index}:s"
            dest = f"{e.dest.id.hex}:i{e.dest_index}:n"
            dot.edge(source, dest)

        end_label = (
            f"{{ {{ {'|'.join(f'<i{i}> {i}' for i in range(len(output_edges)))} }} | end }}"
        )
        dot.node("end", label=end_label)
        for i, e in enumerate(output_edges):
            source = f"{e.source.id.hex}:o{e.source_index}:s"
            dest = f"end:i{i}:n"
            dot.edge(source, dest)
        display.display_svg(dot)

    def add_to_dot(self, dot):
        """
        Recursively adds composite stages to a dot diagram which is being
        constructed.
        Does not add the edges.

        Parameters
        ----------
        dot : graphviz.Diagraph
            dot instance to add edges to.
        """
        with dot.subgraph(name=f"cluster_{uuid4().hex}") as subg:
            subg.attr(color="grey")
            subg.attr(fontcolor="grey")
            if self._name:
                subg.attr(label=self._name)
            for n in self._stages:
                n.add_to_dot(subg)
            for composite_stage in self._composite_stages:
                composite_stage.add_to_dot(subg)
