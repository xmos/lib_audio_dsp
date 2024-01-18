
from typing import Iterable
from pathlib import Path
from .graph import Graph
from .stage import StageOutput
from .thread import Thread
import graphviz
from IPython import display
import yaml
import subprocess
from uuid import uuid4


class Pipeline:
    """
    Top level class which is a container for a list of threads that
    are connected in series.
    """

    def __init__(self, n_in, fs=48000):
        self.graph = Graph()
        self._threads = []
        self.n_in = n_in
        self.n_out = self.n_in
        self.frame_size = 1

        self.i = [StageOutput(fs=fs) for _ in range(n_in)]
        for i, input in enumerate(self.i):
            self.graph.add_edge(input)
            input.source_index = i
        self.o = None

    def add_thread(self):
        ret = Thread(id=len(self._threads), graph=self.graph)
        self._threads.append(ret)
        return ret

    def set_outputs(self, output_edges):
        """set the pipeline outputs, configures the output channel index"""
        for i, edge in enumerate(output_edges):
            edge.dest_index = i

    def validate(self):
        """pipeline must be straight with no branches"""
        graphdict = {}
        for edge in self.graph.edges:
            try:
                graphdict[edge.dest].append(edge.source)
            except KeyError:
                graphdict[edge.dest] = [edge.source]
        for dests in graphdict.values():
            for a, b in zip(dests[:-1], dests[1:]):
                if a is not b:
                    raise ValueError("pipeline must be linear")

    def draw(self):
        """Render a dot diagram of this pipeline"""
        dot = graphviz.Digraph()
        dot.clear()
        for thread in self._threads:
            thread.add_to_dot(dot)
        for e in self.graph.edges:
            source = e.source.id.hex if e.source is not None else "start"
            dest = e.dest.id.hex if e.dest is not None else "end"
            if e.dest is None and e.dest_index is None:
                # unconnected
                dest = uuid4().hex
                dot.node(dest, "", shape="point")
            dot.edge(source, dest, taillabel=str(e.source_index), headlabel=str(e.dest_index))
        display.display_svg(dot)

    def resolve_pipeline(self):
        """
        Generate a dictionary with all of the information about the thread.
        Actual stage instances not included.
        """
        # 1. Order the graph
        sorted_nodes = self.graph.sort()

        # 2. assign nodes to threads
        threads = [[] for _ in range(len(self._threads))]
        for i, thread in enumerate(self._threads):
            for node in sorted_nodes:
                if thread.contains_stage(node):
                    threads[i].append([node.index, node.name])


        edges = []
        for edge in self.graph.edges:
            source = [edge.source.index, edge.source_index] if edge.source is not None else [None, edge.source_index]
            dest = [edge.dest.index, edge.dest_index] if edge.dest is not None else [None, edge.dest_index]
            edges.append([source, dest])

        node_configs = {node.index: node.get_config() for node in self.graph.nodes}

        module_definitions = {node.name: node.yaml_dict for node in self.graph.nodes}

        return {
            "threads": threads, 
            "edges": edges, 
            "configs": node_configs, 
            "modules": module_definitions
        }

def send_config_to_device(pipeline: Pipeline, host_app = "xvf_host", protocol="usb"):
    config = pipeline.resolve_pipeline()["configs"]
    for instance, instance_config in config.items():
        for command, value in instance_config.items():
            if isinstance(value, list) or isinstance(value, tuple):
                value = " ".join(str(v) for v in value)
            print(host_app, "--use", protocol, "--instance-id", str(instance),
                            command, *value.split())
            subprocess.run([host_app, "--use", protocol, "--instance-id", str(instance),
                            command, *value.split()])

def filter_edges_by_thread(resolved_pipeline):
    """get thread input edges, output edges and internal edges for all threads"""
    dest_in_thread = lambda edge, thread: edge[1][0] in (t[0] for t in thread)
    source_in_thread = lambda edge, thread: edge[0][0] in (t[0] for t in thread)
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


def generate_dsp_threads(resolved_pipeline, block_size = 1):
    """
    void dsp_thread(chanend_t* input_c, chanend_t* output_c, module_states, module_configs) {
        int32_t edge0[BLOCK_SIZE];
        int32_t edge1[BLOCK_SIZE];
        int32_t edge2[BLOCK_SIZE];
        int32_t edge3[BLOCK_SIZE];
        for(;;) {
            // input from 2 source threads
            int read_count = 2;
            while(read_count) {
                select {
                    input_c[0]: chan_in_buf(edge0); read_count--;
                    input_c[1]: chan_in_buf(edge1); read_count--;
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
    all_thread_edges = filter_edges_by_thread(resolved_pipeline)
    file_str = ""
    for thread_index, (thread_edges, thread) in enumerate(zip(all_thread_edges, resolved_pipeline["threads"])):
        func = f"DECLARE_JOB(dsp_thread{thread_index}, (chanend_t*, chanend_t*, module_instance_t**));\n"
        func += f"void dsp_thread{thread_index}(chanend_t* c_source, chanend_t* c_dest, module_instance_t** modules) {{\n"

        in_edges, internal_edges, all_output_edges, dead_edges = thread_edges
        all_edges = []
        for temp_in_e in in_edges.values():
            all_edges.extend(temp_in_e)
        all_edges.extend(internal_edges)
        for temp_out_e in all_output_edges.values():
            all_edges.extend(temp_out_e)
        all_edges.extend(dead_edges)
        for i in range(len(all_edges)):
            func += f"\tint32_t edge{i}[{block_size}];\n"
        func += "\twhile(1) {\n"

        # Each thread must process the pending control requests at least once per loop.
        # It will be done once before select to ensure it happens, then in the default
        # case of the select so that control will be processed if no audio is playing.
        control = ""
        for i in range(len(thread)):
            control += f"\t\tmodules[{i}]->module_control(modules[{i}]->state, &modules[{i}]->control);\n"

        func += control
        func += f"\tint read_count = {len(in_edges)};\n"  # TODO use bitfield and guarded cases to prevent
                                                          # the same channel being read twice
        if len(in_edges.values()):
            func += "\tSELECT_RES(\n"
            for i, _ in enumerate(in_edges.values()):
                func += f"\t\tCASE_THEN(c_source[{i}], case_{i}),\n"
            func += "\t\tDEFAULT_THEN(do_control)\n"
            func += "\t) {\n"

            for i, edges in enumerate(in_edges.values()):
                func += f"\t\tcase_{i}: {{\n"
                for edge in edges:
                    func += f"\t\t\tchan_in_buf_word(c_source[{i}], (void*)edge{all_edges.index(edge)}, {block_size});\n"
                func += f"\t\t\tif(!--read_count) break;\n\t\t\telse continue;\n\t\t}}\n"
            func += "\t\tdo_control: {\n"
            func += control
            func += "\t\tcontinue; }\n"
            func += "\t}\n"


        for stage_thread_index, stage in enumerate(thread):
            # thread stages are already ordered during pipeline resolution
            input_edges = [edge for edge in all_edges if edge[1][0] == stage[0]]
            input_edges.sort(key = lambda e: e[1][1])
            input_edges = ", ".join(f"edge{all_edges.index(e)}" for e in input_edges)
            output_edges = [edge for edge in all_edges if edge[0][0] == stage[0]]
            output_edges.sort(key = lambda e: e[0][1])
            output_edges = ", ".join(f"edge{all_edges.index(e)}" for e in output_edges)

            func += f"\tmodules[{stage_thread_index}]->process_sample(\n"
            func += f"\t\t(int32_t*[]){{{input_edges}}},\n"
            func += f"\t\t(int32_t*[]){{{output_edges}}},\n"
            func += f"\t\tmodules[{stage_thread_index}]->state);\n"

        for out_index, edges in enumerate(all_output_edges.values()):
            for edge in edges:
                func += f"\tchan_out_buf_word(c_dest[{out_index}], (void*)edge{all_edges.index(edge)}, {block_size});\n"

        func += "\t}\n}\n"
        file_str += func
    return file_str

def determine_channels(resolved_pipeline):
    """
    create list of the required channels from the resolved pipeline structure.
    """
    all_thread_edges = filter_edges_by_thread(resolved_pipeline)
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

def resolved_pipeline_num_modules(resolved_pipeline):
    return sum(len(t) for t in resolved_pipeline["threads"])

def generate_dsp_struct(resolved_pipeline):

    channels = []
    for c_source, c_dest in determine_channels(resolved_pipeline):
        channels.append(f"channel_{c_source}_{c_dest}")

    struct = "struct audio_dsp_impl {\n\t"
    struct += "\n\t".join(f"channel_t {channel};" for channel in channels)
    struct += "\n"

    num_modules = resolved_pipeline_num_modules(resolved_pipeline)
    struct += f"\tmodule_instance_t* modules[{num_modules}];\n" 
    struct += f"\tint num_modules;\n" 
    struct += "};"
    return struct

def generate_dsp_header(resolved_pipeline, out_dir = "build/dsp_pipeline"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    header = "#pragma once\n"
    header += "#include <xccompat.h>\n"
    header += "#include <xcore/channel.h>\n"
    header += "#include <adsp_module.h>\n"
    header += generate_dsp_struct(resolved_pipeline)

    (out_dir / "adsp_generated.h").write_text(header)

def generate_dsp_init(resolved_pipeline):
    """
    create the init function
    """
    chans = determine_channels(resolved_pipeline)

    ret = "\nvoid adsp_pipeline_init(audio_dsp_t* adsp) {\n"

    for chan_s, chan_d in chans:
        ret += f"\tadsp->channel_{chan_s}_{chan_d} = chan_alloc();\n"

    num_modules = resolved_pipeline_num_modules(resolved_pipeline)
    ret += f"\tadsp->num_modules = {num_modules};\n"

    # initialise the modules
    for thread in resolved_pipeline["threads"]:
        for stage_index, stage_name in thread:
            stage_n_in = len([e for e in resolved_pipeline["edges"] if e[1][0] == stage_index])
            stage_n_out = len([e for e in resolved_pipeline["edges"] if e[0][0] == stage_index])
            stage_frame_size = 1 # TODO

            ret += f"\tadsp->modules[{stage_index}] = {stage_name}_init({stage_index}, {stage_n_in}, {stage_n_out}, {stage_frame_size}, NULL);\n"

    ret += "}\n\n"
    return ret

def generate_dsp_source(resolved_pipeline):
    """
    Generate a function which will be called by the data source, 
    it receives the new samples as input and sends them over the 
    correct channels
    """
    all_edges = filter_edges_by_thread(resolved_pipeline)
    all_in_edges = [e[0] for e in all_edges]

    ret = "void adsp_pipeline_source(audio_dsp_t* adsp, int32_t** data) {\n"

    for dest_thread, thread_input_edges in enumerate(all_in_edges):
        try:
            edges = thread_input_edges["pipeline_in"]
            for edge in edges:
                frame_size = 1  # TODO
                ret += f"\tchan_out_buf_word(adsp->channel_pipeline_in_{dest_thread}.end_a, (uint32_t*)data[{edge[0][1]}], {frame_size});\n"
        except KeyError:
            pass  # this thread doesn't consume from the pipeline input
    ret += "}\n\n"
    return ret

def generate_dsp_sink(resolved_pipeline):
    """
    Generate a function that will be called by the consumer of 
    the dsp output. It will read from the output channels. This function
    will not read from the channels until data is available to allow for 
    the case where 
    """
    all_edges = filter_edges_by_thread(resolved_pipeline)
    all_out_edges = [e[2] for e in all_edges]

    # function to check if channel is not empty
    ret = """
static bool check_chanend(chanend_t c) {
    SELECT_RES(CASE_THEN(c, has_data), DEFAULT_THEN(no_data)) {
        has_data: return true;
        no_data: return false;
    }
}

"""


    sink_details = []
    for source_thread, thread_output_edges in enumerate(all_out_edges):
        try:
            edges = thread_output_edges["pipeline_out"]
            for edge in edges:
                frame_size = 1  # TODO
                sink_details.append((f"adsp->channel_{source_thread}_pipeline_out.end_b", edge[1][1], frame_size))
        except KeyError:
            pass  # this thread doesn't consume from the pipeline input
    
    chanends = set(c for c, _, _ in sink_details)
    ret += "bool adsp_pipeline_sink_nowait(audio_dsp_t* adsp, int32_t** data) {\n"
    ret += f"\tif({' && '.join(f'check_chanend({c})' for c in chanends)}) {{\n"
    for chan, data_idx, frame_size in sink_details:
        ret += f"\t\tchan_in_buf_word({chan}, (uint32_t*)data[{data_idx}], {frame_size});\n"
    ret += "\t\treturn true;\n\t} else { return false; }\n"
    ret += "}\n\n"

    ret += "void adsp_pipeline_sink(audio_dsp_t* adsp, int32_t** data) {\n"
    for chan, data_idx, frame_size in sink_details:
        ret += f"\tchan_in_buf_word({chan}, (uint32_t*)data[{data_idx}], {frame_size});\n"
    ret += "}\n\n"
    return ret

def generate_dsp_main(pipeline: Pipeline, out_dir = "build/dsp_pipeline"):
    """
    Generate the sourcecode for dsp_main

    TODO -  needs to support parallel threads i.e. each
    thread can talk to more than one other thread.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    resolved_pipe = pipeline.resolve_pipeline()

    generate_dsp_header(resolved_pipe)
    threads = resolved_pipe["threads"]

    n_threads = len(threads)

    chans = determine_channels(resolved_pipe)

    dsp_main = """
#include "adsp_pipeline.h"
#include "dspt_main.h"
#include <xcore/select.h>
#include <xcore/channel.h>
#include <print.h>
"""
    dsp_main += generate_dsp_threads(resolved_pipe)
    dsp_main += generate_dsp_source(resolved_pipe)
    dsp_main += generate_dsp_sink(resolved_pipe)

    dsp_main += generate_dsp_init(resolved_pipe)
    dsp_main += "void adsp_pipeline_main(audio_dsp_t* adsp) {\n"

    all_thread_edges = filter_edges_by_thread(resolved_pipe)
    for thread_idx, (thread, thread_edges) in enumerate(zip(threads, all_thread_edges)):
        thread_input_edges, _, thread_output_edges, _ = thread_edges
        # thread stages
        dsp_main += f"\tmodule_instance_t* thread_{thread_idx}_modules[] = {{\n"
        for stage_idx, _ in thread:
            dsp_main += f"\t\tadsp->modules[{stage_idx}],\n"
        dsp_main += "\t};\n"
        # thread input chanends
        input_channels = ",\n\t\t".join([f"adsp->channel_{source}_{thread_idx}.end_b" for source in thread_input_edges.keys()])
        dsp_main += f"\tchanend_t thread_{thread_idx}_inputs[] = {{\n\t\t{input_channels}}};\n"
        # thread output chanends
        output_channels = ",\n\t\t".join([f"adsp->channel_{thread_idx}_{dest}.end_a" for dest in thread_output_edges.keys()])
        dsp_main += f"\tchanend_t thread_{thread_idx}_outputs[] = {{\n\t\t{output_channels}}};\n"

    dsp_main += "\tPAR_JOBS(\n\t\t"
    dsp_main += ",\n\t\t".join(f"PJOB(dsp_thread{ti}, (thread_{ti}_inputs, thread_{ti}_outputs, thread_{ti}_modules))" for ti in range(len(threads)))
    dsp_main += "\n\t);\n"

    dsp_main += "}\n"

    (out_dir / "dsp_main.c").write_text(dsp_main)

    yaml_dir = out_dir/"yaml"
    yaml_dir.mkdir(exist_ok=True)

    for name, defintion in resolved_pipe["modules"].items():
        (yaml_dir / f"{name}.yaml").write_text(yaml.dump(defintion))
