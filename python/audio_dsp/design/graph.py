# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Basic data structures for managing the pipeline graph."""

from uuid import uuid4
import graphlib
from typing import Generic, TypeVar


class Node:
    """
    Graph node.

    Attributes
    ----------
    id : uuid.UUID4
        A unique identifier for this node.
    index : None | int
        node index in the graph. This is set by Graph
        when it is added to the graph.
    """

    def __init__(self):
        self.id = uuid4()
        self.index: int | None = None

    def __hash__(self) -> int:
        """Support for using as dictionary/set keys."""
        return self.id.int


class Edge:
    """
    Graph node.

    Attributes
    ----------
    id : uuid.UUID4
        A unique identifier for this node.
    source : Node | None
    dest : Node | None
        source and dest are the graph nodes that this edge connects between.
    """

    def __init__(self):
        self.id = uuid4()
        self.source: None | Node = None
        self.dest: None | Node = None

    def __hash__(self) -> int:
        """Support for using as dictionary/set keys."""
        return self.id.int

    def set_source(self, node: Node):
        self.source = node

    def set_dest(self, node: Node):
        self.dest = node


NodeSubClass = TypeVar("NodeSubClass", bound=Node)


class Graph(Generic[NodeSubClass]):
    def __init__(self):
        self.nodes: list[NodeSubClass] = []
        self.edges: list[Edge] = []
        self._locked = False

    def add_node(self, node: NodeSubClass) -> None:
        assert isinstance(node, Node)
        if self._locked:
            raise RuntimeError("Cannot add nodes to a locked graph")
        node.index = len(self.nodes)
        self.nodes.append(node)

    def add_edge(self, edge) -> None:
        if self._locked:
            raise RuntimeError("Cannot add edges to a locked graph")
        self.edges.append(edge)

    def get_view(self, nodes: list[NodeSubClass]) -> "Graph[NodeSubClass]":
        """
        Get a filtered view of the graph, including only the provided nodes and the
        edges which connect to them.
        """
        ret = Graph()
        ret.nodes = nodes
        ret.edges = [e for e in self.edges if e.source in nodes or e.dest in nodes]
        return ret

    def get_dependency_dict(self) -> dict[NodeSubClass, set[NodeSubClass]]:
        """
        Return a mapping of nodes to their dependencies ready for use with the graphlib
        utilities.
        """
        graph = {}
        for node in self.nodes:
            graph[node] = set()

        for edge in self.edges:
            if edge.dest in graph and edge.source is not None:
                graph[edge.dest].add(edge.source)
        return graph

    def sort(self) -> tuple[NodeSubClass, ...]:
        """
        Sort the nodes in the graph based on the order they should be executed.
        This is determined by looking at the edges in the graph and resolving the
        order.

        Returns
        -------
        tuple[Node]
            Ordered list of nodes
        """
        return tuple(graphlib.TopologicalSorter(self.get_dependency_dict()).static_order())

    def lock(self):
        """
        Lock the graph. Adding nodes or edges to a locked graph will cause a runtime exception.
        The graph is locked once the pipeline checksum is computed.
        """
        self._locked = True
