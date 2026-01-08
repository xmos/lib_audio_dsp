# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""some graphviz helpers."""

import graphviz  # type: ignore


def new_record_digraph():
    """
    Create a digraph with some attributes set.

    Returns
    -------
    graphviz.Digraph
        A graph object
    """
    dot = graphviz.Digraph()
    dot.clear()
    dot.attr(ranksep="1.0")
    dot.attr("node", shape="record")
    return dot
