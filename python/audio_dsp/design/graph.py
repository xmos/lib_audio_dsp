
from uuid import uuid4
import graphlib

class Node:
    def __init__(self, obj):
        self.id = uuid4()
        self.obj = obj
        self.index = None

    def __hash__(self) -> int:
        return self.id.int

class Edge:
    def __init__(self):
        self.id = uuid4()
        self.source = None
        self.dest = None

    def __hash__(self) -> int:
        return self.id.int

    def set_source(self, node: Node):
        self.source = node

    def set_dest(self, node: Node):
        self.dest = node

class Graph:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def add_node(self, node: Node) -> None:
        assert isinstance(node, Node)
        node.index = len(self.nodes)
        self.nodes.append(node)

    def add_edge(self, edge) -> None:
        self.edges.append(edge)

    def sort(self):
        """returns ordered list of nodes"""
        graph = {}
        for edge in self.edges:
            if edge.dest is not None:
                try:
                    graph[edge.dest].add(edge.source)
                except KeyError:
                    graph[edge.dest] = set((edge.source,))

        return tuple(graphlib.TopologicalSorter(graph).static_order())


