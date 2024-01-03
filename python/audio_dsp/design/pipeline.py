
from typing import Iterable
from pathlib import Path
from .graph import Graph
from .stage import StageOutput
from .thread import Thread
import graphviz
from IPython import display
import yaml
import subprocess


class Pipeline:
    """
    Top level class which is a container for a list of threads that
    are connected in series.
    """

    def __init__(self, n_in):
        self.graph = Graph()
        self._threads = []

        self.i = [StageOutput() for _ in range(n_in)]
        for i, input in enumerate(self.i):
            self.graph.add_edge(input)
            input.source_index = i
        self.o = None

    def add_thread(self):
        ret = Thread(id=len(self._threads), graph=self.graph)
        self._threads.append(ret)
        return ret

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
        for i_thread, thread in enumerate(self._threads):
            with dot.subgraph(name=f"cluster_{i_thread}") as subg:
                subg.attr(label=f"thread {i_thread}")
                for i, n in enumerate(self.graph.nodes):
                    if thread.contains_stage(n):
                        subg.node(n.id.hex, f"{i}: {type(n).__name__}")
        for e in self.graph.edges:
            source = e.source.id.hex if e.source is not None else "start"
            dest = e.dest.id.hex if e.dest is not None else "end"
            dot.edge(source, dest)
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

        return {"threads": threads, "edges": edges, "configs": node_configs, "modules": module_definitions}

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


def generate_dsp_main(pipeline: Pipeline, out_dir = "build/dsp_pipeline"):
    """
    Generate the sourcecode for dsp_main

    TODO -  needs to support parallel threads i.e. each
    thread can talk to more than one other thread.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    resolved_pipe = pipeline.resolve_pipeline()
    threads = resolved_pipe["threads"]

    n_threads = len(threads)

    chans = ["chan_in", *(f"chan_{i}" for i in range(n_threads))]

    dsp_main = """
#include "dspt_main.h"
#pragma stackfunction 1000
void dspt_xcore_main(chanend_t c_data, chanend_t c_control)
{
"""
    for chan in chans:
        dsp_main += f"channel_t {chan} = chan_alloc();\n"

    total_modules = 0
    for i, thread in enumerate(threads):
        n_stages = len(thread)
        total_modules += n_stages
        dsp_main += f"const int32_t num_modules_thread{i} = {n_stages};\n"

    dsp_main += f"int total_num_modules = {total_modules};\n"
    dsp_main += f"module_instance_t* all_modules[{total_modules}];\n\n"

    for i, thread in enumerate(threads):
        dsp_main += f"module_instance_t* modules{i}[] = {{\n"
        for mod_i, mod_name in thread:
            dsp_main += f"\t{mod_name}_init({mod_i}),\n"
        dsp_main += f"}};\n\n"
        for this_i, (mod_i, mod_name) in enumerate(thread):
            dsp_main += f"all_modules[{mod_i}] = modules{i}[{this_i}];\n"


    dsp_main += f"""
     PAR_JOBS(
        PJOB(dsp_data_transport_thread, (c_data, {chans[0]}.end_a, {chans[-1]}.end_b)),
        PJOB(dsp_control_thread, (c_control, all_modules, total_num_modules))"""

    for i, (chan_a, chan_b) in enumerate(zip(chans[:-1], chans[1:])):
        dsp_main += f",\n        PJOB(dsp_thread, ({chan_a}.end_b, {chan_b}.end_a, modules{i}, num_modules_thread{i}))"

    dsp_main += "\n    );\n}\n"

    (out_dir / "dsp_main.c").write_text(dsp_main)

    yaml_dir = out_dir/"yaml"
    yaml_dir.mkdir(exist_ok=True)

    for name, defintion in resolved_pipe["modules"].items():
        (yaml_dir / f"{name}.yaml").write_text(yaml.dump(defintion))
