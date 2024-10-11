from ..design._draw import new_record_digraph
from uuid import uuid4


def _add_stage(dot, ir, stage_name, stage):
    inputs = ir.get_node_inputs(stage_name)
    n_in = 0 if not inputs else max(i.destination.index for i in inputs.values())
    outputs = ir.get_node_outputs(stage_name)
    n_out = 0 if not outputs else max(i.source.index for i in outputs.values())
    center = stage_name
    inputs = "|".join(f"<i{i}> " for i in range(n_in + 1))
    outputs = "|".join(f"<o{i}> " for i in range(n_out + 1))
    label = f"{{ {{ {inputs} }} | {center} | {{ {outputs} }}}}"
    dot.node(stage_name, label)


def _add_buffers(dot, ir):
    for name, buffer in ir.config_struct.buffers.items():
        center = f"{name} {buffer.ratio.input}:{buffer.ratio.output}"
        label = center
        dot.node(name, label=label)


def _add_threads(dot, ir):
    for thread_id, stage_dict in ir.split_stages_by_thread().items():
        with dot.subgraph(name=f"cluster_{uuid4().hex}") as subg:
            subg.attr(color="grey")
            subg.attr(fontcolor="grey")
            subg.attr(label=f"Thread {thread_id}")

            for stage_name, stage in stage_dict.items():
                _add_stage(subg, ir, stage_name, stage)


def _add_start_end(dot, ir):
    input_indices = ir.input_indices()
    output_indices = ir.output_indices()

    start_label = (
        f"{{ start | {{ {'|'.join(f'<o{i}> {i}' for i in range(max(input_indices) + 1))} }} }}"
    )
    end_label = (
        f"{{ {{ {'|'.join(f'<i{i}> {i}' for i in range(max(output_indices)+1))} }} | end }}"
    )
    dot.node("start", label=start_label)
    dot.node("end", label=end_label)


def _edge_label(edge):
    return "\n".join(f"{k}: {v}" for k, v in edge.type.shape.model_dump().items())


def _add_edges(dot, ir):
    for edge in ir.config_struct.edges:
        source_node = "start" if edge.dsp_input else edge.source.name
        dest_node = "end" if edge.dsp_output else edge.destination.name
        if source_node in ir.config_struct.buffers:
            source = source_node
        else:
            source = f"{source_node}:o{edge.source.index}:s"  #  "s" means connect to the "south" of the port
        if dest_node in ir.config_struct.buffers:
            dest = dest_node
        else:
            dest = f"{dest_node}:i{edge.destination.index}:n"  #  "n" means connect to the "north" of the port
        dot.edge(
            source, dest, xlabel=_edge_label(edge)
        )  # xlabel doesn't impact the layout, just draws the label on top


def _gen_dot(ir):
    dot = new_record_digraph()

    _add_start_end(dot, ir)
    _add_buffers(dot, ir)
    _add_threads(dot, ir)
    _add_edges(dot, ir)

    return dot


def render(ir, path):
    dot = _gen_dot(ir)
    dot.format = "png"
    dot.render(path)
