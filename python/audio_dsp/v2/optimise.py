from .types import UnknownEdge


class InputUnknownError(Exception):
    pass


class NodeMultipleInputsError(Exception):
    pass


class IncompatibleEdgeError(Exception):
    pass


class BufferWithNoInputsError(Exception):
    pass


class UnresolvableGraphError(Exception):
    pass


def _inputs_have_type_and_shape(ir):
    """Check the graphs inputs all have a type and shape."""
    graph_inputs = {i: e for i, e in enumerate(ir.config_struct.edges) if e.dsp_input}
    for i, e in graph_inputs.items():
        if e.type == "unknown":
            raise InputUnknownError(f"edge[{i}] is an input, its type must be known!")


def _test_edge_is_expected_or_unknown(edge, expected_type):
    if edge.type.name == "unknown":
        return True
    if edge.type != expected_type:
        raise IncompatibleEdgeError
    return False  # test passed, no change required


def _test_stage_output_expected_or_unknown(outputs, expected):
    """Called for each output index of a stage, determine if an output needs updating"""
    made_change = False
    if isinstance(expected, UnknownEdge):
        return False  # test not done

    for output in outputs:
        if isinstance(output.type, UnknownEdge):
            output.type = expected
            made_change = True
        else:  # output type already set
            if output.type != expected:
                raise IncompatibleEdgeError
    return made_change


def _resolve_edges(ir):
    """
    Determine the edge type, shape, config for all edges, raising errors where stages are connected
    to incompatible edges. Or where edges are connected to incompatible stages.
    """
    # A DSP graph is a DAG that flows from Buffers to Buffers. A buffers output type is known if its
    # input type is known.
    # Start byt resolving the output of all buffers, then resolve all stages. then try again.

    # set containing the names of buffers and stages who's output edges are all okay.
    tested_buffers = set()
    tested_stages = set()

    while True:
        made_change = False
        for name, buffer in ir.config_struct.buffers.items():
            buf_in = ir.get_node_inputs(name)
            if not buf_in:
                raise BufferWithNoInputsError
            buf_out = ir.get_node_outputs(name)
            assert buf_out, "Buffers should already have been given outputs if they were missing"
            if len(buf_in) > 1:
                raise NodeMultipleInputsError(
                    f"Buffers can only have 1 input, {name} has {len(buf_in)}"
                )
            if len(buf_out) > 1:
                raise NotImplementedError("TODO support buffers with multiple output edges")
            buf_in_i, buf_in_conf = next(iter(buf_in.items()))
            buf_input_type = buf_in_conf.type.name
            buf_out_i, buf_out_conf = next(iter(buf_out.items()))
            buf_output_type = (
                buf_out_conf.type.name
            )  # all buffer outputs should have the same type
            if name not in tested_buffers and buf_input_type != "unknown":
                edge_type = ir.config_struct.load_edge(buf_input_type)
                expected_out = edge_type.buffer_transform(ir.edge_context(buf_in_i), buffer)
                if _test_edge_is_expected_or_unknown(buf_out_conf, expected_out):
                    buf_out_conf.type = expected_out
                    made_change = True
                # test has passed if the input and output buffers are not unknown
                tested_buffers.add(name)

        for name, stage in ir.config_struct.stages.items():
            # a stage has multiple inputs
            stage_in = ir.get_node_inputs(name)
            stage_out = ir.get_node_outputs(name)
            if name not in tested_stages:
                stage_type = ir.config_struct.load_stage(stage.type)
                expected_outputs = stage_type.determine_outputs(
                    ir.stage_context(name), [s.type for s in stage_in.values()]
                )
                for i, expected_type in enumerate(expected_outputs):
                    # Each output index can have multiple edges
                    outputs_i = [e for e in stage_out.values() if e.source.index == i]
                    made_change = (
                        _test_stage_output_expected_or_unknown(outputs_i, expected_type)
                        or made_change
                    )
                if all(not isinstance(e.type, UnknownEdge) for e in stage_out.values()) and all(
                    not isinstance(e.type, UnknownEdge) for e in stage_in.values()
                ):
                    tested_stages.add(name)
        if not made_change:
            # A complete pass of all the stages and buffers occured without any more type
            # information being added to the graph. Edge resolution is complete.
            break

    unknown_edges = [e for e in ir.config_struct.edges if isinstance(e.type, UnknownEdge)]
    if unknown_edges:
        raise UnresolvableGraphError(
            f"The following edge types could not be resolved: "
            + ", ".join(u.repr_route() for u in unknown_edges)
        )

    return ir


def _add_missing_buffer_outputs(ir):
    """All buffers have one output, if one is missing from the ir, add it with an unknown type."""
    for name, buffer in ir.config_struct.buffers.items():
        if not ir.get_node_outputs(name):
            ir.config_struct.edges.append(
                ir.edge_model(type={"name": "unknown"}, source={"name": name, "index": 0})
            )
    return ir


def _add_missing_stage_outputs(ir):
    """
    All stage outputs need an edge to ensure that the generated code contains
    memory for that output - even if the output is not consumed the stage
    will still need to create it.
    """
    return ir


def validate_input(ir):
    """
    Check graph validity by queryng all contained stages and edges. Adding resolved information
    to the graph so it is ready to be transformed into the next layer.
    """
    _inputs_have_type_and_shape(ir)
    # TODO
    # - each stage input has only 1 edge connects to it
    # - Each buffer has 1 input and output
    # - all edges connect to valid stages
    # - all output edges have unique indices (input edges may go to multiple buffers)
    # - All stages on the same thread have the same execution rate. (need to understand how to solve)
    # - All threads are internally acyclic

    ir = _add_missing_buffer_outputs(ir.copy())
    ir = _add_missing_stage_outputs(ir.copy())
    ir = _resolve_edges(ir.copy())

    # at this point we have a valid input graph which is ready to be translated to the
    # next layer of IR.
    return ir


def optimise(ir):
    """
    This function turns a validated input into an optimised next level of IR. This level
    will have all edge aliasing removed, ensuring that edges which share memory will be
    consumed safely in the generated code.

    This is the first pass of optimisation and performs the following steps:

    1. Transform input into a valid IR - This will involve inserting copy stages
       onto every edge which aliases another, and every edge which shares a stage output.
    2. in a loop -
        a. validate the IR
        b. optmisation pass
    """
    return ir
