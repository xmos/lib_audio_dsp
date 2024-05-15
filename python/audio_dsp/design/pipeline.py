# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Top level pipeline design class and code generation functions."""

from pathlib import Path
from tabulate import tabulate

from audio_dsp.design.pipeline_executor import PipelineExecutor, PipelineView
from .graph import Graph
from .stage import Stage, StageOutput, find_config
from .thread import Thread
from IPython import display
import yaml
import hashlib
import json
import numpy as np
from uuid import uuid4
from ._draw import new_record_digraph
from .host_app import send_control_cmd
from functools import wraps
from typing import NamedTuple


class _ResolvedEdge(NamedTuple):
    """Resolved representation of an edge, used in code gen."""

    source: tuple[int, int]
    dest: tuple[int, int]
    fs: float
    frame_size: int


def callonce(f):
    """Decorate functions to ensure they only execute once despite being called multiple times."""
    attr_name = "_called_funcs"

    def called_funcs_of_instance(instance) -> set:
        called_funcs = getattr(instance, attr_name, set())
        if not called_funcs:
            setattr(instance, attr_name, called_funcs)
        return called_funcs

    @wraps(f)
    def wrapper(*args, **kwargs):
        self = args[0]
        called_funcs = called_funcs_of_instance(self)

        if f not in called_funcs:
            called_funcs.add(f)
            wrapper.called = True
            return f(*args, **kwargs)

    return wrapper


class PipelineStage(Stage):
    """
    Stage for the pipelne. Does not support processing of data through it. Only
    used for pipeline level control commands, for example, querying the pipeline
    checksum.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("pipeline"), **kwargs)
        self.create_outputs(0)

    def add_to_dot(self, dot):  # Override this to not add the stage to the diagram
        """
        Override the CompositeStage.add_to_dot() function to ensure PipelineStage
        type stages are not added to the dot diagram.

        Parameters
        ----------
        dot : graphviz.Diagraph
        dot instance to add edges to.
        """
        return


class Pipeline:
    """
    Top level class which is a container for a list of threads that
    are connected in series.

    Parameters
    ----------
    n_in : int
        Number of input channels into the pipeline
    identifier: string
        Unique identifier for this pipeline. This identifier will be included in
        the generated header file name (as "adsp_generated_<identifier>.h"), the
        generated source file name (as "adsp_generated_<identifier>.c"), and the
        pipeline's generated initialisation and main functions (as
        "adsp_<identifier>_pipeline_init" and
        "adsp_<identifier>_pipeline_main")
    frame_size : int
        Size of the input frame of all input channels
    fs : int
        Sample rate of the input channels

    Attributes
    ----------
    i : list(StageOutput)
        The inputs to the pipeline should be passed as the inputs to the
        first stages in the pipeline
    threads : list(Thread)
        List of all the threads in the pipeline
    pipeline_stage : PipelineStage | None
        Stage corresponding to the pipeline. Needed for handling
        pipeline level control commands
    """

    def __init__(self, n_in, identifier="auto", frame_size=1, fs=48000):
        self._graph = Graph()
        self.threads = []
        self._n_in = n_in
        self._n_out = 0
        self._id = identifier
        self.pipeline_stage: None

        self.i = [StageOutput(fs=fs, frame_size=frame_size) for _ in range(n_in)]
        self.o: list[StageOutput] | None = None
        for i, input in enumerate(self.i):
            self._graph.add_edge(input)
            input.source_index = i

    def add_thread(self):
        """
        Create a new instance of audio_dsp.thread.Thread and add it to
        the pipeline. Stages can then be instantiated in the thread.

        Returns
        -------
        Thread
            a new thread instance.
        """
        ret = Thread(id=len(self.threads), graph=self._graph)

        self.add_pipeline_stage(ret)

        ret.add_thread_stage()
        self.threads.append(ret)
        return ret

    @callonce
    def add_pipeline_stage(self, thread):
        """Add a PipelineStage stage for the pipeline."""
        self.pipeline_stage = thread.stage(PipelineStage, [])

    def set_outputs(self, output_edges: list[StageOutput]):
        """
        Set the pipeline outputs, configures the output channel index.

        Parameters
        ----------
        output_edges : Iterable(None | StageOutput)
            configure the output channels and their indices. Outputs of the pipeline
            will be in the same indices as the input to this function. To have an empty
            output index, pass in None.
        """
        if not output_edges:
            raise RuntimeError("Pipeline must have at least 1 output")
        i = -1
        for i, edge in enumerate(output_edges):
            if edge is not None:
                edge.dest_index = i
        self.o = output_edges
        self._n_out = i + 1
        self.resolve_pipeline()  # Call it here to generate the pipeline hash

    def executor(self) -> PipelineExecutor:
        """Create an executor instance which can be used to simulate the pipeline."""

        def view():
            if self.o is None:
                raise RuntimeError(
                    "Pipeline outputs must be set with `set_outputs` before simulating"
                )
            return PipelineView(self._graph.nodes, self.i, self.o)

        return PipelineExecutor(self._graph, view)

    def validate(self):
        """
        TODO validate pipeline assumptions.

        - Thread connections must not lead to a scenario where the pipeline hangs
        - Stages must fit on thread
        - feedback must be within a thread (future)
        - All edges have the same fs and frame_size (until future enhancements)
        """

    def draw(self):
        """Render a dot diagram of this pipeline."""
        dot = new_record_digraph()
        for thread in self.threads:
            thread.add_to_dot(dot)
        start_label = f"{{ start | {{ {'|'.join(f'<o{i}> {i}' for i in range(self._n_in))} }} }}"
        end_label = f"{{ {{ {'|'.join(f'<i{i}> {i}' for i in range(self._n_out))} }} | end }}"
        dot.node("start", label=start_label)
        dot.node("end", label=end_label)
        for e in self._graph.edges:
            source = e.source.id.hex if e.source is not None else "start"
            source = (
                f"{source}:o{e.source_index}:s"  #  "s" means connect to the "south" of the port
            )
            dest = e.dest.id.hex if e.dest is not None else "end"
            dest = f"{dest}:i{e.dest_index}:n"  #  "n" means connect to the "north" of the port
            if e.dest is None and e.dest_index is None:
                # unconnected
                dest = uuid4().hex
                dot.node(dest, "", shape="point")
            dot.edge(source, dest)
        display.display_svg(dot)

    @property
    def stages(self):
        """Flattened list of all the stages in the pipeline."""
        return self._graph.nodes[:]

    @callonce
    def generate_pipeline_hash(self, threads: list, edges: list):
        """
        Generate a hash unique to the pipeline and save it in the 'checksum' control field of the
        pipeline stage.

        Parameters
        ----------
        "threads": list of [[(stage index, stage type name), ...], ...] for all threads in the pipeline
        "edges": list of [[[source stage, source index], [dest stage, dest index]], ...] for all edges in the pipeline
        """

        def to_tuple(lst):
            return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

        tuple_threads = to_tuple(threads)
        tuple_edges = to_tuple(edges)
        tuple_thread_edges = (tuple_threads, tuple_edges)
        a = (json.dumps(tuple_thread_edges)).encode()

        m = hashlib.md5()
        m.update(a)
        hash_a = m.digest()

        hash_values = [i for i in bytearray(hash_a)]

        assert self.pipeline_stage is not None  # To stop ruff from complaining

        self.pipeline_stage["checksum"] = hash_values
        # lock the graph now that the hash is generated
        self._graph.lock()

    def resolve_pipeline(self):
        """
        Generate a dictionary with all of the information about the thread.
        Actual stage instances not included.

        Returns
        -------
        dict
            'identifier': string identifier for the pipeline
            "threads": list of [[(stage index, stage type name, stage memory use), ...], ...] for all threads
            "edges": list of [[[source stage, source index], [dest stage, dest index]], ...] for all edges
            "configs": list of dicts containing stage config for each stage.
            "modules": list of stage yaml configs for all types of stage that are present
            "labels": dictionary {label: instance_id} defining mapping between the user defined stage labels and the index of the stage
        """
        # 1. Order the graph
        sorted_nodes = self._graph.sort()

        # 2. assign nodes to threads
        threads = [[] for _ in range(len(self.threads))]
        for i, thread in enumerate(self.threads):
            for node in sorted_nodes:
                if thread.contains_stage(node):
                    threads[i].append([node.index, node.name, node.get_required_allocator_size()])

        edges = []
        for edge in self._graph.edges:
            source = (
                [edge.source.index, edge.source_index]
                if edge.source is not None
                else [None, edge.source_index]
            )
            dest = (
                [edge.dest.index, edge.dest_index]
                if edge.dest is not None
                else [None, edge.dest_index]
            )
            edges.append(_ResolvedEdge(source, dest, frame_size=edge.frame_size, fs=edge.fs))

        self.generate_pipeline_hash(threads, edges)

        node_configs = {node.index: node.get_config() for node in self._graph.nodes}

        module_definitions = {
            node.index: {"name": node.name, "yaml_dict": node.yaml_dict}
            for node in self._graph.nodes
        }

        labels = {node.label: node.index for node in self._graph.nodes if node.label is not None}

        return {
            "identifier": self._id,
            "threads": threads,
            "edges": edges,
            "configs": node_configs,
            "modules": module_definitions,
            "labels": labels,
        }


def validate_pipeline_checksum(pipeline: Pipeline):
    """
    Check if python and device pipeline checksums match. Raise a runtime error if the checksums are not equal.

    Parameters
    ----------
    pipeline : Python pipeline for which to validate checksum against the device pipeline
    """
    assert pipeline.pipeline_stage is not None  # To stop ruff from complaining

    ret = send_control_cmd(pipeline.pipeline_stage.index, "pipeline_checksum")

    if ret.returncode:
        raise RuntimeError("Unable to connect to device using host app")

    stdout = ret.stdout.decode().splitlines()
    device_pipeline_checksum = [int(x) for x in stdout]
    equal = np.array_equal(
        np.array(device_pipeline_checksum), np.array(pipeline.pipeline_stage["checksum"])
    )

    if equal is False:
        raise RuntimeError(
            f"Python pipeline checksum {pipeline.pipeline_stage['checksum']} does not match device pipeline checksum {device_pipeline_checksum}"
        )


def send_config_to_device(pipeline: Pipeline):
    """
    Send the current config for all stages to the device.
    Make sure set_host_app() is called before calling this to set a valid host app.

    Parameters
    ----------
    pipeline : Pipeline
        A designed and optionally tuned pipeline
    """
    validate_pipeline_checksum(pipeline)

    for stage in pipeline.stages:
        for command, value in stage.get_config().items():
            command = f"{stage.name}_{command}"
            if isinstance(value, list) or isinstance(value, tuple):
                value = " ".join(str(v) for v in value)
            else:
                value = str(value)

            ret = send_control_cmd(stage.index, command, *value.split(), verbose=True)

            if ret.returncode:
                return


def _filter_edges_by_thread(resolved_pipeline):
    """
    Get thread input edges, output edges and internal edges for all threads.

    Returns
    -------
    list[tuple[]]
        input_edges, internal_edges, output_edges, dead_edges for all threads in
        the resolved pipeline.
        input_edges and output_edges are dictionaries of {source_or_dest_thread: [edges]}
        internal and dead edges are list of edges.
    """

    def dest_in_thread(edge, thread):
        return edge[1][0] in (t[0] for t in thread)

    def source_in_thread(edge, thread):
        return edge[0][0] in (t[0] for t in thread)

    ret = []

    for thread in resolved_pipeline["threads"]:
        input_edges = {}
        output_edges = {}
        internal_edges = []
        dead_edges = []
        for edge in resolved_pipeline["edges"]:
            sit = source_in_thread(edge, thread)
            dit = dest_in_thread(edge, thread)
            if sit and dit:
                internal_edges.append(edge)
            elif (edge[1][0] is None and edge[1][1] is None) and (dit or sit):
                # This edge is not connected to an output
                dead_edges.append(edge)
            elif sit:
                if edge[1][0] is None:
                    # pipeline output
                    di = "pipeline_out"
                else:
                    for di, dthread in enumerate(resolved_pipeline["threads"]):
                        if dest_in_thread(edge, dthread):
                            break
                try:
                    output_edges[di].append(edge)
                except KeyError:
                    output_edges[di] = [edge]
            elif dit:
                if edge[0][0] is None:
                    # pipeline input
                    si = "pipeline_in"
                else:
                    for si, sthread in enumerate(resolved_pipeline["threads"]):
                        if source_in_thread(edge, sthread):
                            break
                try:
                    input_edges[si].append(edge)
                except KeyError:
                    input_edges[si] = [edge]
        ret.append((input_edges, internal_edges, output_edges, dead_edges))
    return ret


def _gen_chan_buf_read_q31_to_q27(channel, edge, frame_size):
    """Generate the C code to read from a channel and convert from q31 to q27."""
    return f"chan_in_buf_word({channel}, (uint32_t*){edge}, {frame_size}); for(int idx = 0; idx < {frame_size}; ++idx) {edge}[idx] = adsp_from_q31({edge}[idx]);"


def _gen_chan_buf_write_q27_to_q31(channel, edge, frame_size):
    """Generate the C code to write to a channel and convert from q27 to q31 and saturate."""
    return f"for(int idx = 0; idx < {frame_size}; ++idx) {edge}[idx] = adsp_to_q31({edge}[idx]); chan_out_buf_word({channel}, (uint32_t*){edge}, {frame_size});"


def _generate_dsp_threads(resolved_pipeline):
    """
    Create the source string for all of the dsp threads.

    Output looks approximately like the below::

        void dsp_thread(chanend_t* input_c, chanend_t* output_c, module_states, module_configs) {
            int32_t edge0[BLOCK_SIZE];
            int32_t edge1[BLOCK_SIZE];
            int32_t edge2[BLOCK_SIZE];
            int32_t edge3[BLOCK_SIZE];
            for(;;) {
                do control;
                // input from 2 source threads
                int read_count = 2;
                while(read_count) {
                    select {
                        input_c[0]: chan_in_buf(edge0); read_count--;
                        input_c[1]: chan_in_buf(edge1); read_count--;
                        default: do control;
                    }
                }
                modules[0]->process_sample(
                    (int32_t*[]){edge0, edge1},
                    (int32_t*[]){edge2, edge3},
                    modules[0]->state,
                    &modules[0]->control
                );
                chan_out_buf(output_c[0], edge2);
                chan_out_buf(output_c[1], edge3);
            }
        }
    """
    all_thread_edges = _filter_edges_by_thread(resolved_pipeline)
    file_str = ""
    for thread_index, (thread_edges, thread) in enumerate(
        zip(all_thread_edges, resolved_pipeline["threads"])
    ):
        func = f"DECLARE_JOB(dsp_{resolved_pipeline['identifier']}_thread{thread_index}, (chanend_t*, chanend_t*, module_instance_t**));\n"
        func += f"void dsp_{resolved_pipeline['identifier']}_thread{thread_index}(chanend_t* c_source, chanend_t* c_dest, module_instance_t** modules) {{\n"

        # set high priority thread bit to ensure we get the required
        # MIPS
        func += "\tlocal_thread_mode_set_bits(thread_mode_high_priority);"

        in_edges, internal_edges, all_output_edges, dead_edges = thread_edges
        is_input_thread = "pipeline_in" in in_edges
        all_edges = []
        for temp_in_e in in_edges.values():
            all_edges.extend(temp_in_e)
        all_edges.extend(internal_edges)
        for temp_out_e in all_output_edges.values():
            all_edges.extend(temp_out_e)
        all_edges.extend(dead_edges)
        for i, edge in enumerate(all_edges):
            func += f"\tint32_t edge{i}[{edge.frame_size}] = {{0}};\n"

        # get the dsp_thread stage index in the thread
        dsp_thread_index = [i for i, (_, name, _) in enumerate(thread) if name == "dsp_thread"][0]

        for stage_thread_index, stage in enumerate(thread):
            # thread stages are already ordered during pipeline resolution
            input_edges = [edge for edge in all_edges if edge[1][0] == stage[0]]
            output_edges = [edge for edge in all_edges if edge[0][0] == stage[0]]
            if not (input_edges or output_edges):
                # stages with no inputs and outputs also have no process method
                # so they don't need input and output variables
                continue
            else:
                if len(input_edges) > 0:  # To avoid compilation warnings
                    input_edges.sort(key=lambda e: e[1][1])
                    input_edges = ", ".join(f"edge{all_edges.index(e)}" for e in input_edges)
                    func += f"\tint32_t* stage_{stage_thread_index}_input[] = {{{input_edges}}};\n"
                else:
                    func += f"\tint32_t** stage_{stage_thread_index}_input = NULL;\n"

                if len(output_edges) > 0:
                    output_edges.sort(key=lambda e: e[0][1])
                    output_edges = ", ".join(f"edge{all_edges.index(e)}" for e in output_edges)
                    func += (
                        f"\tint32_t* stage_{stage_thread_index}_output[] = {{{output_edges}}};\n"
                    )
                else:
                    func += f"\tint32_t** stage_{stage_thread_index}_output = NULL;\n"

        func += "\tuint32_t start_ts, end_ts, start_control_ts, control_ticks;\n"
        func += "\tbool control_done;\n"

        func += "\twhile(1) {\n"
        func += "\tcontrol_done = false;\n"

        # Each thread must process the pending control requests at least once per loop.
        # It will be done once before select to ensure it happens, then in the default
        # case of the select so that control will be processed if no audio is playing.
        control = ""
        for i, (stage_index, name, _) in enumerate(thread):
            if resolved_pipeline["modules"][stage_index]["yaml_dict"]:
                control += f"\t\t{name}_control(modules[{i}]->state, &modules[{i}]->control);\n"

        read = f"\tint read_count = {len(in_edges)};\n"  # TODO use bitfield and guarded cases to prevent
        # the same channel being read twice
        if len(in_edges.values()):
            read += "\tSELECT_RES(\n"
            for i, _ in enumerate(in_edges.values()):
                read += f"\t\tCASE_THEN(c_source[{i}], case_{i}),\n"
            read += "\t\tDEFAULT_THEN(do_control)\n"
            read += "\t) {\n"

            for i, (origin, edges) in enumerate(in_edges.items()):
                read += f"\t\tcase_{i}: {{\n"
                for edge in edges:
                    if origin == "pipeline_in":
                        read += (
                            "\t\t\t"
                            + _gen_chan_buf_read_q31_to_q27(
                                f"c_source[{i}]", f"edge{all_edges.index(edge)}", edge.frame_size
                            )
                            + "\n"
                        )
                    else:
                        read += f"\t\t\tchan_in_buf_word(c_source[{i}], (void*)edge{all_edges.index(edge)}, {edge.frame_size});\n"
                read += "\t\t\tif(!--read_count) break;\n\t\t\telse continue;\n\t\t}\n"
            read += "\t\tdo_control: {\n"
            read += "\t\tstart_control_ts = get_reference_time();\n"
            read += control
            read += "\t\tcontrol_done = true;\n"
            read += "\t\tcontrol_ticks = get_reference_time() - start_control_ts;\n"
            read += "\t\tcontinue; }\n"
            read += "\t}\n"

        read += "\tif(!control_done){\n"
        read += "\t\tstart_control_ts = get_reference_time();\n"
        read += control
        read += "\t\tcontrol_ticks = get_reference_time() - start_control_ts;\n"
        read += "\t}\n"

        process = "\tstart_ts = get_reference_time();\n\n"

        for stage_thread_index, (stage_index, name, _) in enumerate(thread):
            input_edges = [edge for edge in all_edges if edge[1][0] == stage_index]
            output_edges = [edge for edge in all_edges if edge[0][0] == stage_index]

            # thread stages are already ordered during pipeline resolution
            if len(input_edges) > 0 or len(output_edges) > 0:
                process += f"\t{name}_process(\n"
                process += f"\t\tstage_{stage_thread_index}_input,\n"
                process += f"\t\tstage_{stage_thread_index}_output,\n"
                process += f"\t\tmodules[{stage_thread_index}]->state);\n"

        process += "\n\tend_ts = get_reference_time();\n"

        profile = "\tuint32_t process_plus_control_ticks = (end_ts - start_ts) + control_ticks;\n"
        profile += f"\tif(process_plus_control_ticks > ((dsp_thread_state_t*)(modules[{dsp_thread_index}]->state))->max_cycles)\n"
        profile += "\t{\n"
        profile += f"\t\t((dsp_thread_state_t*)(modules[{dsp_thread_index}]->state))->max_cycles = process_plus_control_ticks;\n"
        profile += "\t}\n"

        out = ""
        for out_index, (dest, edges) in enumerate(all_output_edges.items()):
            for edge in edges:
                if dest == "pipeline_out":
                    out += (
                        "\t"
                        + _gen_chan_buf_write_q27_to_q31(
                            f"c_dest[{out_index}]", f"edge{all_edges.index(edge)}", edge.frame_size
                        )
                        + "\n"
                    )
                else:
                    out += f"\tchan_out_buf_word(c_dest[{out_index}], (void*)edge{all_edges.index(edge)}, {edge.frame_size});\n"

        # The pipeline start condition must be that it is already full so reads
        # can be done without worrying about synchronisation. This is done
        # by starting on a read for the input threads, and starting on an
        # out for the other threads
        if is_input_thread:
            file_str += func + read + process + profile + out + "\t}\n}\n"
        else:
            file_str += func + out + read + process + profile + "\t}\n}\n"

    return file_str


def _determine_channels(resolved_pipeline):
    """Create list of the required channels from the resolved pipeline structure."""
    all_thread_edges = _filter_edges_by_thread(resolved_pipeline)
    ret = []
    for s_idx, s_thread_edges in enumerate(all_thread_edges):
        s_in_edges, _, _, _ = s_thread_edges
        # add pipeline entry channels
        if "pipeline_in" in s_in_edges.keys():
            ret.append(("pipeline_in", s_idx))
    for s_idx, s_thread_edges in enumerate(all_thread_edges):
        _, _, s_out_edges, _ = s_thread_edges
        ret.extend((s_idx, dest) for dest in s_out_edges.keys())
    return ret


def _resolved_pipeline_num_modules(resolved_pipeline):
    """Determine total number of module instances in the resolved pipeline across all threads."""
    return sum(len(t) for t in resolved_pipeline["threads"])


def _generate_dsp_header(resolved_pipeline, out_dir=Path("build/dsp_pipeline")):
    """Generate "adsp_generated_<x>.h" and save to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    header = "#pragma once\n"
    header += "#include <stages/adsp_pipeline.h>\n"
    header += "\n"
    header += f"adsp_pipeline_t * adsp_{resolved_pipeline['identifier']}_pipeline_init();\n"
    header += (
        f"void adsp_{resolved_pipeline['identifier']}_pipeline_main(adsp_pipeline_t* adsp);\n"
    )

    (out_dir / f"adsp_generated_{resolved_pipeline['identifier']}.h").write_text(header)


def _generate_instance_id_defines(resolved_pipeline, out_dir=Path("build/dsp_pipeline")):
    """Generate "adsp_instance_id.h" that defines the stage indexes for stages labelled by the user and save to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    pipeline_id = resolved_pipeline["identifier"]
    n_threads = len(resolved_pipeline["threads"])

    header = "#pragma once\n\n"
    for label, index in resolved_pipeline["labels"].items():
        header += f"#define {label}_stage_index\t\t({index})\n"

    thread_stage_ids = ", ".join(f"thread{i}_stage_index" for i in range(n_threads))
    header += f"#define {pipeline_id}_thread_stage_indices  {{ {thread_stage_ids} }}\n"
    (out_dir / f"adsp_instance_id_{pipeline_id}.h").write_text(header)


def _generate_dsp_init(resolved_pipeline):
    """Create the init function which initialised all modules and channels."""
    chans = _determine_channels(resolved_pipeline)
    adsp = f"adsp_{resolved_pipeline['identifier']}"

    ret = f"adsp_pipeline_t * {adsp}_pipeline_init() {{\n"
    ret += f"\tstatic adsp_pipeline_t {adsp};\n\n"

    # Track the number of channels so we can initialise them
    input_channels = 0
    link_channels = 0
    output_channels = 0

    for chan_s, chan_d in chans:
        # We assume that there is no channel pipeline_in -> pipeline_out
        if chan_s == "pipeline_in":
            input_channels += 1
        elif chan_d == "pipeline_out":
            output_channels += 1
        else:
            link_channels += 1

    ret += f"\tstatic channel_t {adsp}_in_chans[{input_channels}];\n"
    ret += f"\tstatic channel_t {adsp}_out_chans[{output_channels}];\n"
    ret += f"\tstatic channel_t {adsp}_link_chans[{link_channels}];\n"

    num_modules = _resolved_pipeline_num_modules(resolved_pipeline)
    ret += f"\tstatic module_instance_t {adsp}_modules[{num_modules}];\n"

    # We assume that this function generates the arrays adsp_<x>_(in|out)_mux_cfgs
    # and that it will initialise the .(input|output)_mux members of adsp
    ret += _generate_dsp_muxes(resolved_pipeline)

    for chan in range(input_channels):
        ret += f"\t{adsp}_in_chans[{chan}] = chan_alloc();\n"
    for chan in range(output_channels):
        ret += f"\t{adsp}_out_chans[{chan}] = chan_alloc();\n"
    for chan in range(link_channels):
        ret += f"\t{adsp}_link_chans[{chan}] = chan_alloc();\n"
    ret += f"\t{adsp}.p_in = (channel_t *) {adsp}_in_chans;\n"
    ret += f"\t{adsp}.n_in = {input_channels};\n"
    ret += f"\t{adsp}.p_out = (channel_t *) {adsp}_out_chans;\n"
    ret += f"\t{adsp}.n_out = {output_channels};\n"
    ret += f"\t{adsp}.p_link = (channel_t *) {adsp}_link_chans;\n"
    ret += f"\t{adsp}.n_link = {link_channels};\n"
    ret += f"\t{adsp}.modules = {adsp}_modules;\n"
    ret += f"\t{adsp}.n_modules = {num_modules};\n"

    # initialise the modules
    for thread in resolved_pipeline["threads"]:
        for stage_index, stage_name, stage_mem in thread:
            in_edges = [e for e in resolved_pipeline["edges"] if e.dest[0] == stage_index]
            out_edges = [e for e in resolved_pipeline["edges"] if e.source[0] == stage_index]
            stage_n_in = len(in_edges)
            stage_n_out = len(out_edges)
            # TODO this is naive, stage could have different frame sizes on each edge
            try:
                stage_frame_size = max(e.frame_size for e in in_edges + out_edges)
            except ValueError:
                # the stage has no edges
                stage_frame_size = 1

            defaults = {}
            for config_field, value in resolved_pipeline["configs"][stage_index].items():
                if isinstance(value, list) or isinstance(value, tuple):
                    defaults[config_field] = "{" + ", ".join(str(i) for i in value) + "}"
                else:
                    defaults[config_field] = str(value)
            struct_val = ", ".join(f".{field} = {value}" for field, value in defaults.items())
            # default_str = f"&({stage_name}_config_t){{{struct_val}}}"
            if resolved_pipeline["modules"][stage_index]["yaml_dict"]:
                ret += (
                    f"\tstatic {stage_name}_config_t config{stage_index} = {{ {struct_val} }};\n"
                )

            ret += f"""
            static {stage_name}_state_t state{stage_index};
            static uint8_t memory{stage_index}[{stage_mem}];
            static adsp_bump_allocator_t allocator{stage_index} = ADSP_BUMP_ALLOCATOR_INITIALISER(memory{stage_index});

            {adsp}.modules[{stage_index}].state = (void*)&state{stage_index};

            // Control stuff
            {adsp}.modules[{stage_index}].control.id = {stage_index};
            {adsp}.modules[{stage_index}].control.config_rw_state = config_none_pending;
            """
            if resolved_pipeline["modules"][stage_index]["yaml_dict"]:
                ret += f"""
                {adsp}.modules[{stage_index}].control.config = (void*)&config{stage_index};
                {adsp}.modules[{stage_index}].control.module_type = e_dsp_stage_{stage_name};
                {adsp}.modules[{stage_index}].control.num_control_commands = NUM_CMDS_{stage_name.upper()};
                """
            else:
                ret += f"""
                {adsp}.modules[{stage_index}].control.config = NULL;
                {adsp}.modules[{stage_index}].control.num_control_commands = 0;
                """

            ret += f"{stage_name}_init(&{adsp}.modules[{stage_index}], &allocator{stage_index}, {stage_index}, {stage_n_in}, {stage_n_out}, {stage_frame_size});\n"
    ret += f"\treturn &{adsp};\n"
    ret += "}\n\n"
    return ret


def _generate_dsp_muxes(resolved_pipeline):
    # We assume that this function generates the arrays adsp_<x>_(in|out)_mux_cfgs
    # and that it will initialise the .(input|output)_mux members of adsp
    # We assume that dictionaries are ordered (which is true as of Python 3.7)

    all_edges = _filter_edges_by_thread(resolved_pipeline)
    all_in_edges = [e[0] for e in all_edges]
    all_out_edges = [e[2] for e in all_edges]
    adsp = f"adsp_{resolved_pipeline['identifier']}"

    # We're basically assuming here that the dictionary is ordered the same way
    # as it's going to be when we construct main, so these thread relationships
    # are always going to be the same. I think this is true.

    input_chan_idx = 0
    num_input_mux_cfgs = 0
    ret = f"\tstatic adsp_mux_elem_t {adsp}_in_mux_cfgs[] = {{\n"
    for thread_input_edges in all_in_edges:
        try:
            edges = thread_input_edges["pipeline_in"]
            for edge in edges:
                ret += f"\t\t{{ .channel_idx = {input_chan_idx}, .data_idx = {edge[0][1]}, .frame_size = {edge.frame_size}}},\n"
                num_input_mux_cfgs += 1
            input_chan_idx += 1
        except KeyError:
            pass  # this thread doesn't consume from the pipeline input
    ret += "\t};\n"

    output_chan_idx = 0
    num_output_mux_cfgs = 0
    ret += f"\tstatic adsp_mux_elem_t {adsp}_out_mux_cfgs[] = {{\n"
    for thread_output_edges in all_out_edges:
        try:
            edges = thread_output_edges["pipeline_out"]
            for edge in edges:
                ret += f"\t\t{{ .channel_idx = {output_chan_idx}, .data_idx = {edge[1][1]}, .frame_size = {edge.frame_size}}},\n"
                num_output_mux_cfgs += 1
            output_chan_idx += 1
        except KeyError:
            pass  # this thread doesn't consume from the pipeline input
    ret += "\t};\n"

    ret += f"\t{adsp}.input_mux.n_chan = {num_input_mux_cfgs};\n"
    ret += f"\t{adsp}.input_mux.chan_cfg = (adsp_mux_elem_t *) {adsp}_in_mux_cfgs;\n"
    ret += f"\t{adsp}.output_mux.n_chan = {num_output_mux_cfgs};\n"
    ret += f"\t{adsp}.output_mux.chan_cfg = (adsp_mux_elem_t *) {adsp}_out_mux_cfgs;\n"

    return ret


def generate_dsp_main(pipeline: Pipeline, out_dir="build/dsp_pipeline"):
    """
    Generate the source code for adsp_generated_<x>.c.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to generate code for.
    out_dir : str
        Directory to store generated code in.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    resolved_pipe = pipeline.resolve_pipeline()

    _generate_instance_id_defines(resolved_pipe, out_dir)

    _generate_dsp_header(resolved_pipe, out_dir)
    threads = resolved_pipe["threads"]

    dsp_main = """
#include <stages/adsp_pipeline.h>
#include <xcore/select.h>
#include <xcore/channel.h>
#include <xcore/hwtimer.h>
#include <xcore/thread.h>
#include <print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include <stages/bump_allocator.h>
#include <dsp/signal_chain.h>

"""
    # add includes for each stage type in the pipeline
    dsp_main += "".join(
        f"#include <stages/{resolved_pipe['modules'][node_index]['name']}.h>\n"
        for node_index in resolved_pipe["modules"].keys()
    )
    dsp_main += "".join(
        f"#include <{resolved_pipe['modules'][node_index]['name']}_config.h>\n"
        for node_index in resolved_pipe["modules"].keys()
        if resolved_pipe["modules"][node_index]["yaml_dict"] is not None
    )
    dsp_main += _generate_dsp_threads(resolved_pipe)
    dsp_main += _generate_dsp_init(resolved_pipe)

    dsp_main += (
        f"void adsp_{resolved_pipe['identifier']}_pipeline_main(adsp_pipeline_t* adsp) {{\n"
    )

    input_chan_idx = 0
    output_chan_idx = 0
    all_thread_edges = _filter_edges_by_thread(resolved_pipe)
    determined_channels = _determine_channels(resolved_pipe)
    for thread_idx, (thread, thread_edges) in enumerate(zip(threads, all_thread_edges)):
        thread_input_edges, _, thread_output_edges, _ = thread_edges
        # thread stages
        dsp_main += f"\tmodule_instance_t* thread_{thread_idx}_modules[] = {{\n"
        for stage_idx, _, _ in thread:
            dsp_main += f"\t\t&adsp->modules[{stage_idx}],\n"
        dsp_main += "\t};\n"

        # thread input chanends
        input_channel_array = []
        for source in thread_input_edges.keys():
            if source == "pipeline_in":
                array_source = "adsp->p_in"
                idx_num = input_chan_idx
                input_chan_idx += 1
            else:
                array_source = "adsp->p_link"
                link_idx = 0
                for chan_s, chan_d in determined_channels:
                    if chan_s != "pipeline_in" and chan_d != "pipeline_out":
                        if source == chan_s and thread_idx == chan_d:
                            idx_num = link_idx
                            break
                        else:
                            link_idx += 1
                else:
                    raise RuntimeError("Channel not found")
            input_channel_array.append(f"{array_source}[{idx_num}].end_b")
        input_channels = ",\n\t\t".join(input_channel_array)
        dsp_main += f"\tchanend_t thread_{thread_idx}_inputs[] = {{\n\t\t{input_channels}}};\n"

        # thread output chanends
        output_channel_array = []
        for dest in thread_output_edges.keys():
            if dest == "pipeline_out":
                array_source = "adsp->p_out"
                idx_num = output_chan_idx
                output_chan_idx += 1
            else:
                array_source = "adsp->p_link"
                link_idx = 0
                for chan_s, chan_d in determined_channels:
                    if chan_s != "pipeline_in" and chan_d != "pipeline_out":
                        if chan_s == thread_idx and chan_d == dest:
                            idx_num = link_idx
                            break
                        else:
                            link_idx += 1
                else:
                    raise RuntimeError("Channel not found")
            output_channel_array.append(f"{array_source}[{idx_num}].end_a")
        output_channels = ",\n\t\t".join(output_channel_array)
        dsp_main += f"\tchanend_t thread_{thread_idx}_outputs[] = {{\n\t\t{output_channels}}};\n"

    dsp_main += "\tPAR_JOBS(\n\t\t"
    dsp_main += ",\n\t\t".join(
        f"PJOB(dsp_{resolved_pipe['identifier']}_thread{ti}, (thread_{ti}_inputs, thread_{ti}_outputs, thread_{ti}_modules))"
        for ti in range(len(threads))
    )
    dsp_main += "\n\t);\n"

    dsp_main += "}\n"

    (out_dir / f"adsp_generated_{resolved_pipe['identifier']}.c").write_text(dsp_main)

    yaml_dir = out_dir / "yaml"
    yaml_dir.mkdir(exist_ok=True)

    for name, defintion in resolved_pipe["modules"].items():
        (yaml_dir / f"{name}.yaml").write_text(yaml.dump(defintion))


def profile_pipeline(pipeline: Pipeline):
    """
    Profiles the DSP threads that are a part of the pipeline.
    Make sure set_host_app() is called before calling this to set a valid host app.

    Parameters
    ----------
    pipeline : Pipeline
        A designed and optionally tuned pipeline
    """
    validate_pipeline_checksum(pipeline)

    # print("Thread Index     Max Cycles")
    profile_info = []
    for thread in pipeline.threads:
        thread_fs = None
        thread_frame_size = None
        stages = thread.get_all_stages()
        for stg in stages:
            if stg.fs is not None:
                thread_fs = stg.fs
                thread_frame_size = stg.frame_size
                break
        # Assuming that all stages in the thread have the same sampling freq and frame size
        if thread_fs is None:
            raise RuntimeError(
                f"Could not find out the sampling frequency for thread index {thread.id}"
            )

        if thread_frame_size is None:
            raise RuntimeError(f"Could not find out the frame size for thread index {thread.id}")

        reference_timer_freq_hz = 100e6
        frame_time_s = float(thread_frame_size) / thread_fs
        ticks_per_sample_time_s = reference_timer_freq_hz * frame_time_s
        ticks_per_sample_time_s = ticks_per_sample_time_s

        # TODO Implement a generic way of reading all config from the stage
        command = "dsp_thread_max_cycles"
        ret = send_control_cmd(thread.thread_stage.index, command)

        if ret.returncode:
            return

        cycles = int(ret.stdout.splitlines()[0].decode("utf-8"))
        percentage_used = (cycles / ticks_per_sample_time_s) * 100
        profile_info.append(
            [thread.id, round(ticks_per_sample_time_s, 2), cycles, round(percentage_used, 2)]
        )
    print(
        tabulate(
            profile_info,
            headers=[
                "thread index",
                "available time (ref timer ticks)",
                "max ticks consumed",
                "% consumed",
            ],
            tablefmt="pretty",
        )
    )
