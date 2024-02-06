
"""
Basic data structures for managing the pipeline graph.
"""
from uuid import uuid4
import graphlib

class Node:
    """
    Graph node

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
        self.index = None

    def __hash__(self) -> int:
        """Support for using as dictionary/set keys"""
        return self.id.int

class Edge:
    """
    Graph node

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
        self.source = None
        self.dest = None

    def __hash__(self) -> int:
        """Support for using as dictionary/set keys"""
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
        """
        Sort the nodes in the graph based on the order they should be executed.
        This is determined by looking at the edges in the graph and resolving the
        order.

        Returns
        -------
        tuple[Node]
            Ordered list of nodes
        """
        graph = {}
        for node in self.nodes:
            graph[node] = set()

        for edge in self.edges:
            if edge.dest is not None:
                try:
                    graph[edge.dest].add(edge.source)
                except KeyError:
                    graph[edge.dest] = set((edge.source,))

        return tuple(graphlib.TopologicalSorter(graph).static_order())


