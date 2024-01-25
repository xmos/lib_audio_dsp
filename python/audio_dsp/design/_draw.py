"""
some graphviz helpers.
"""
import graphviz

def new_record_digraph():
    """
    Create a digraph with some attributes set

    Returns
    -------
    graphviz.Digraph
        A graph object
    """
    dot = graphviz.Digraph()
    dot.clear()
    dot.attr(fontname="Arial Nova Light")
    dot.attr('node', shape='record')
    dot.attr('node', fontname='Arial Nova Light')
    dot.attr('edge', fontname='Arial Nova Light')
    return dot