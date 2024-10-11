from pathlib import Path
from mako.template import Template
from types import SimpleNamespace
from graphlib import TopologicalSorter
from .types import BufferType, EdgeConnection

THIS_DIR = Path(__file__).parent
DSP_TEMPLATE = Template(filename=str(THIS_DIR / "code_gen/dsp_generated.mako"))


def _resolve_threads(ir):
    """much of this logic might make more sense in optimise.py"""
    thread_indices = set()
    for stage in ir.config_struct.stages.values():
        thread_indices.add(stage.thread)

    def make_thread():
        return SimpleNamespace(
            input_buffers={},
            output_buffers={},
            stages={},
            edges={},
            stage_order=[],
            unique_edges={},
        )

    threads = {i: make_thread() for i in thread_indices}

    # filter stages
    for name, stage in ir.config_struct.stages.items():
        threads[stage.thread].stages[name] = stage.model_copy()

    # filter edges
    for thread in threads.values():
        for edge_idx, edge in enumerate(ir.config_struct.edges):
            if edge.source.name in thread.stages or edge.destination.name in thread.stages:
                thread.edges[edge_idx] = edge.model_copy()
                # edges with the same source get dropped so edge stage output
                # will have a "unique" edge in the generated code.
                thread.unique_edges[edge.source] = edge.model_copy()

    # find input/output buffers
    for thread in threads.values():
        thread_source_names = [e.source.name for e in thread.edges.values()]
        thread.input_buffers = {
            n: (b.model_copy(), ir.get_node_outputs(n))
            for n, b in ir.config_struct.buffers.items()
            if n in thread_source_names
        }
        thread_destination_names = [e.destination.name for e in thread.edges.values()]
        thread.output_buffers = {
            n: (b.model_copy(), ir.get_node_inputs(n))
            for n, b in ir.config_struct.buffers.items()
            if n in thread_destination_names
        }

    # resolve stage order
    for thread in threads.values():
        graph = {
            name: [
                e.source.name
                for e in thread.edges.values()
                if e.destination.name == name and e.source.name in thread.stages
            ]
            for name in thread.stages
        }
        thread.stage_order = list(TopologicalSorter(graph).static_order())

    return threads


def _edge_var(edge_source):
    return f"edge_{edge_source.name}_{edge_source.index}"


def _buffer_var(name):
    return f"buffer_{name}"


def _stage_var(name):
    return f"stage_{name}"


def _buffer_type_to_prefix(b: BufferType):
    if b == BufferType.ONE_TO_ONE:
        return "buffer_1to1"
    elif b == BufferType.BIG_TO_SMALL:
        return "buffer_bigtosmall"
    else:
        return "buffer_smalltobig"


def _buffer_read(ir, buffer_name, edge_index):
    edge = ir.config_struct.edges[edge_index]
    buffer = ir.config_struct.buffers[buffer_name]
    edge_read, edge_construct = ir.config_struct.load_edge(edge.type.name).c_buffer_read(
        ir.edge_context(edge_index), _edge_var(edge.source)
    )
    buffer_prefix = _buffer_type_to_prefix(buffer.type)
    return [
        f"{buffer_prefix}_read(&{_buffer_var(buffer_name)}, {loc});" for loc, _ in edge_read
    ] + edge_construct


def _dedupe(l):
    """Remove dupes, preserve order"""
    return list(dict.fromkeys(l))


def _buffer_write(ir, buffer_name, edge_index):
    edge = ir.config_struct.edges[edge_index]
    buffer = ir.config_struct.buffers[buffer_name]
    edge_write = ir.config_struct.load_edge(edge.type.name).c_buffer_write(
        ir.edge_context(edge_index), _edge_var(edge.source)
    )
    buffer_prefix = _buffer_type_to_prefix(buffer.type)
    return [f"{buffer_prefix}_write(&{_buffer_var(buffer_name)}, {loc});" for loc, _ in edge_write]


def _stage_process(ir, stage_name):
    input_vars = _dedupe(
        [
            _edge_var(edge.source)
            for i, edge in sorted(
                ir.get_node_inputs(stage_name).items(), key=lambda t: t[1].destination.index
            )
        ]
    )
    output_vars = _dedupe(
        [
            _edge_var(edge.source)
            for i, edge in sorted(
                ir.get_node_outputs(stage_name).items(), key=lambda t: t[1].source.index
            )
        ]
    )
    stage_type = ir.config_struct.stages[stage_name].type
    process_lines = ir.config_struct.load_stage(stage_type).c_process(
        ir.stage_context(stage_name), "&" + _stage_var(stage_name), input_vars, output_vars
    )
    return process_lines


def _create_thread_code_sections(ir, thread):
    ret = SimpleNamespace(
        edge_definitions=[
            ir.config_struct.edge_definitions[edge.type.name]
            .load(edge.type.name)
            .c_definition(ir.edge_context_from_edge(edge), _edge_var(source))
            for source, edge in thread.unique_edges.items()
        ],
        buffer_reads=[
            _buffer_read(ir, b, list(e[1].keys())[0]) for b, e in thread.input_buffers.items()
        ],
        control=[],
        process=[_stage_process(ir, stage) for stage in thread.stage_order],
        buffer_writes=[
            _buffer_write(ir, b, list(e[1].keys())[0]) for b, e in thread.output_buffers.items()
        ],
    )
    return ret


def _edge_sizeof(ir, edge):
    return (
        ir.config_struct.edge_definitions[edge.type.name]
        .load(edge.type.name)
        .c_sizeof(ir.edge_context_from_edge(edge))
    )


def _buffer_definition(ir):
    ret = []
    for name, buffer in ir.config_struct.buffers.items():
        input_edge = next(iter(ir.get_node_inputs(name).values()))
        output_edge = next(iter(ir.get_node_outputs(name).values()))
        in_size = _edge_sizeof(ir, input_edge)
        out_size = _edge_sizeof(ir, output_edge)
        c_buffer_size = f"MAX_({in_size}, {out_size})"
        ret.append(f"static char arr_{_buffer_var(name)}[{c_buffer_size}];")
        bp = _buffer_type_to_prefix(buffer.type)
        ret.append(
            f"static {bp}_t {_buffer_var(name)} = {bp.upper()}_FULL(arr_{_buffer_var(name)}, {in_size}, {out_size});"
        )
    return ret


def _dsp_source(ir):
    input_edges = [e for e in ir.config_struct.edges if e.dsp_input]
    data_param = "data"
    ret = []
    for edge in input_edges:
        buffer_name = edge.destination.name
        buffer = ir.config_struct.buffers[buffer_name]
        bp = _buffer_type_to_prefix(buffer.type)
        # get list of [(pointer, size), ...]
        edge_write = ir.config_struct.load_edge(edge.type.name).c_buffer_write(
            ir.edge_context_from_edge(edge), data_param
        )
        buffer_ops = [f"{bp}_write(&{_buffer_var(buffer_name)}, {loc});" for loc, _ in edge_write]
        ret.append((edge.source.index, buffer_ops))
    return ret


def _dsp_sink(ir):
    output_edges = [e for e in ir.config_struct.edges if e.dsp_output]
    data_param = "data"
    ret = []
    for edge in output_edges:
        buffer_name = edge.source.name
        buffer = ir.config_struct.buffers[buffer_name]
        bp = _buffer_type_to_prefix(buffer.type)
        # get list of [(pointer, size), ...]
        edge_read, edge_reconstruct = ir.config_struct.load_edge(edge.type.name).c_buffer_read(
            ir.edge_context_from_edge(edge), data_param
        )
        buffer_ops = [
            f"{bp}_read(&{_buffer_var(buffer_name)}, {loc});" for loc, _ in edge_read
        ] + edge_reconstruct
        ret.append((edge.destination.index, buffer_ops))
    return ret


def code_gen(ir, out_dir="."):
    out_dir = Path(out_dir)
    auto_gen_file = out_dir / "dsp_autogenerated.c"

    source_cases = _dsp_source(ir)
    sink_cases = _dsp_sink(ir)
    buffer_defs = _buffer_definition(ir)
    thread_graph = _resolve_threads(ir)
    thread_code = [_create_thread_code_sections(ir, t) for t in thread_graph.values()]

    auto_gen_file.write_text(
        DSP_TEMPLATE.render(
            ir=ir,
            threads=thread_code,
            buffer_defs=buffer_defs,
            source_cases=source_cases,
            sink_cases=sink_cases,
        )
    )
