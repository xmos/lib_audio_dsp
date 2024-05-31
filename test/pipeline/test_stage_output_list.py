"""Test that StageOutputList manipulation methods work correctly."""

from audio_dsp.design.stage import StageOutputList, StageOutput
from random import randint

def list_of_edges():
    return [StageOutput() for _ in range(randint(5, 10))]


def test_stage_output_list_basic():
    """Tests StageOutputList creation."""
    input = list_of_edges()
    sol = StageOutputList(input)
    for i, o in zip(input, sol.edges):
        assert i is o

def test_eq():
    """Check that StageOutputList are equal when all contained edges are the same."""
    a = list_of_edges()
    b = list_of_edges()
    assert StageOutputList(a) == StageOutputList(a), "Same edges should be equal"
    assert StageOutputList(a) == StageOutputList(a[:]), "Same edges but different list should be equal"
    assert StageOutputList(a) != StageOutputList(b), "Different edges should be equal"

def test_add():
    """Check that StageOutputList can be combined easily."""
    a = StageOutputList(list_of_edges())
    b = StageOutputList(list_of_edges())

    actual_add = None + a + [None] + b + None
    actual_or = None | a | [None] | b | None

    expected = [None] + a.edges + [None] + b.edges + [None]

    for exp, act_add, act_or in zip(expected, actual_add.edges, actual_or.edges):
        assert exp is act_add
        assert exp is act_or

def test_get():
    """Check that StageOutputList can be indexed by int, tuple, slice, iterator"""
    edges = list_of_edges()
    a = StageOutputList(edges)

    assert a[0] == StageOutputList([edges[0]])
    assert a[0, 2, 4] == StageOutputList([edges[0], edges[2], edges[4]])
    assert a[0:3:2] == StageOutputList([edges[0], edges[2]])
    assert a[range(2)] == StageOutputList([edges[0], edges[1]])





