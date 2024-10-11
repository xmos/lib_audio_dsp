from .types import DspEdge, EdgeShape, EdgeConfig, Buffer
from .ir import EdgeContext

int_edge = DspEdge()

_SIZEOF_INT = 4


@int_edge.shape
class IntShape(EdgeShape):
    n: int = 1
    fs: int = 48000


@int_edge.config
class IntConfig(EdgeConfig):
    is_null: bool = False
    """is_null defines if this edge will be defined as an array or a null pointer."""


@int_edge.buffer_transform
def int_buffer_transform(context: EdgeContext, buffer: Buffer):
    """
    Determine the output type given an input edge and a buffer.

    Parameters
    ----------
        context: dict[]
    """
    if not buffer.ratio.is_integral():
        raise ValueError("Int buffers must have integral ratios")
    n = context.edge.type.shape.n * buffer.ratio_f()
    if n != int(n):
        raise ValueError("incompatible ratio and edge shape")
    return context.edge.type.model_copy(
        update={"shape": IntShape(n=int(n)), "config": IntConfig()}
    )


@int_edge.c_definition
def int_c_definition(context: EdgeContext, edge_var_name: str):
    shape = context.edge.type.shape
    config = context.edge.type.config
    if config.is_null:
        return f"int32_t* {edge_var_name} = NULL;"
    else:
        return f"int32_t {edge_var_name}[{shape.n}];"


@int_edge.c_sizeof
def int_c_sizeof(context: EdgeContext):
    return _SIZEOF_INT * context.edge.type.shape.n


@int_edge.c_buffer_write
def int_c_buffer_write(context: EdgeContext, edge_var_name: str):
    """Return list of (void*, size) that need to be written to the buffer"""
    return [(edge_var_name, f"{context.edge.type.shape.n * _SIZEOF_INT}")]


@int_edge.c_buffer_read
def int_c_buffer_read(context: EdgeContext, edge_var_name: str):
    """
    Get the items to be read from the buffer and also some arbitrary C code to
    reconstruct the edge.

    [(pointer, size), ...], [code_str, ...]
    """
    if context.edge.type.config.is_null:
        raise ValueError("Can't read from a buffer into a null int")
    return [(edge_var_name, f"{context.edge.type.shape.n * _SIZEOF_INT}")], []
