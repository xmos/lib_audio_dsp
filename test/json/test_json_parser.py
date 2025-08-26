# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path
import annotated_types

from audio_dsp.design.parse_json import Graph, make_pipeline, insert_forks, DspJson, pipeline_to_dspjson

from audio_dsp.models.stage import all_models
from audio_dsp.stages import all_stages

from typing import get_origin, get_args, Literal
from types import UnionType


def find_autoforks(graph):
    for node in graph.nodes:
        if 'AutoFork' in node.placement.name:
            return True

    assert False, "No AutoFork node found in the graph after insert_forks."


def test_no_shared_edge():
    json_str = """
    {
      "name": "No shared edge",
      "fs": 44100,
      "inputs": [{
          "name": "inputs",
          "channels": 1
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [["inputs", 0]],
            "name": "VolumeControl_1",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "outputs",
          "input": [["VolumeControl_1", 0]]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)

    a.draw(Path("test_no_shared_edge"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")

    dsp_json = DspJson(ir_version=1, producer_name="pipeline_to_dspjson", producer_version="1.0", graph=new_graph)
    new_json = pipeline_to_dspjson(a)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"

    for node in graph.nodes:
        if 'AutoFork' in node.placement.name:
            assert False, "AutoFork node found in the graph after insert_forks, but not needed."




def test_shared_edge_from_graph_input():
    json_str = """
    {
      "name": "Shared edge from graph input",
      "fs": 44100,
      "inputs": [{
          "name": "inputs",
          "channels": 1
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [["inputs", 0]],
            "name": "VolumeControl_A",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [["inputs", 0]],
            "name": "Mixer_B",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "outputs",
          "input": [["VolumeControl_A", 0]]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)

    a.draw(Path("test_shared_edge_from_graph_input"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")
    find_autoforks(new_graph)


def test_shared_edge_from_producer_node():
    json_str = """
    {
      "name": "Shared edge from producer node",
      "fs": 44100,
      "inputs": [{
          "name": "inputs",
          "channels": 1
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [["inputs", 0]],
            "name": "Producer",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [["Producer", 0]],
            "name": "Consumer_1",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [["Producer", 0]],
            "name": "Consumer_2",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "outputs",
          "input": [["Consumer_1", 0]]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)
    
    a.draw(Path("test_shared_edge_from_producer_node"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")
    find_autoforks(new_graph)


def test_shared_edge_with_graph_output():
    json_str = """
    {
      "name": "Shared edge with graph output",
      "fs": 44100,
      "inputs": [{
          "name": "inputs",
          "channels": 1
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [["inputs", 0]],
            "name": "Producer",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [["Producer", 0]],
            "name": "Consumer",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "outputs",
          "input": [["Producer", 0]]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)
    
    a.draw(Path("test_shared_edge_with_graph_output"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")
    find_autoforks(new_graph)


def test_again():
    json_str = """
    {
      "name": "Stereo Mixer with Volume",
      "fs": 48000,
      "inputs": [{
          "name": "stereo_in",
          "channels": 2
      }],
      "nodes": [
        {"op_type": "Mixer", "placement": {"input": [["stereo_in", 0], ["stereo_in", 1]], "name": "Mixer", "thread": 0}},
        {"op_type": "VolumeControl", "placement": {"input": [["Mixer", 0]], "name": "Volume", "thread": 0}}
      ],
      "outputs": [{
          "name": "stereo_out",
          "input": [["Volume", 0], ["Volume", 0]]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)
    
    a.draw(Path("test_again"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")
    find_autoforks(new_graph)


def test_multiple_inputs_outputs_non_shared():
    """
    Test a graph with two inputs (one mono and one stereo) and two outputs (one mono and one stereo)
    where no edge is shared between consumers.
    """
    json_str = """
    {
      "name": "Multiple Inputs and Outputs Non-Shared Test",
      "fs": 48000,
      "inputs": [
        {"name": "mono_in", "channels": 1},
        {"name": "stereo_in", "channels": 2}
      ],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [["mono_in", 0]],
            "name": "VolumeControl_A",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [["stereo_in", 0], ["stereo_in", 1]],
            "name": "Mixer_B",
            "thread": 0
          }
        }
      ],
      "outputs": [
        {"name": "mono_out", "input": [["VolumeControl_A", 0]]},
        {"name": "stereo_out", "input": [["Mixer_B", 0], ["VolumeControl_A", 0]]}
      ]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)
    
    a.draw(Path("test_multiple_inputs_outputs_non_shared"))
    new_graph = insert_forks(graph)
    print("Test: Multiple Inputs and Outputs Non-Shared Test")
    print(f"Before insert_forks: {graph.model_dump_json(indent=2)}")
    print(f"After insert_forks: {new_graph.model_dump_json(indent=2)}")
    find_autoforks(new_graph)


def test_multiple_inputs_outputs_shared():
    """
    Test a graph with two inputs and two outputs where a node produces an output
    that is shared by both graph outputs. A Fork should be inserted.
    """
    json_str = """
    {
      "name": "Multiple Inputs and Outputs Shared Test",
      "fs": 48000,
      "inputs": [
        {"name": "input1", "channels": 1},
        {"name": "input2", "channels": 1}
      ],
      "nodes": [
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [["input1", 0], ["input2", 0]],
            "name": "Mixer",
            "thread": 0
          }
        }
      ],
      "outputs": [
        {"name": "output1", "input": [["Mixer", 0]]},
        {"name": "output2", "input": [["Mixer", 0]]}
      ]
    }
    """
    graph = Graph.model_validate_json(json_str)
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    a = make_pipeline(dsp_json)
    
    a.draw(Path("test_multiple_inputs_outputs_shared"))
    new_graph = insert_forks(graph)
    print("Test: Multiple Inputs and Outputs Shared Test")
    print(f"Before insert_forks: {graph.model_dump_json(indent=2)}")
    print(f"After insert_forks: {new_graph.model_dump_json(indent=2)}")
    
    find_autoforks(new_graph)





def test_frame_size():
    """
    Test that frame_size in the JSON schema loads/saves correctly into the pipeline object
    """
    json_str = """
    {
      "name": "FrameSizeTest",
      "fs": 48000,
      "frame_size": 8,
      "inputs": [{
          "name": "inputs",
          "channels": 1
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [["inputs", 0]],
            "name": "VolumeControl_1",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "outputs",
          "input": [["VolumeControl_1", 0]]
      }]
    }
    """
    # Parse the JSON and create a pipeline
    graph = Graph.model_validate_json(json_str)
    assert graph.frame_size == 8, "Graph frame_size should be 8"
    
    dsp_json = DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    pipeline = make_pipeline(dsp_json)
    
    # Verify frame_size was set correctly in the pipeline
    assert pipeline.frame_size == 8, "Pipeline frame_size should be 8"
    
    # Convert back to JSON and verify frame_size is preserved
    new_json = pipeline_to_dspjson(pipeline)
    assert new_json.graph.frame_size == 8, "Converted JSON frame_size should be 8"


def test_all_stages_models():
    all_m = all_models()
    all_s = all_stages()

    failed = False
    for s in all_s:
        if s.startswith("_") or s in ["DSPThreadStage", "PipelineStage"]:
            continue

        try:
            assert s in all_m, f"Stage {s} not found in all models"
            assert s == all_m[s].model_fields["op_type"].default, f"Stage {s} op_type mismatch"
        except AssertionError as e:
            print(e)
            failed = True
            continue

        if "biquad" in s.lower() or "parametric" in s.lower():
            continue
        
        if "parameters" not in all_m[s].model_fields:
            continue
        

        try:
            if type(all_s[s].set_parameters.__annotations__["parameters"]) is UnionType:
              set_params_input = get_args(all_s[s].set_parameters.__annotations__["parameters"])
              model_params_type =  all_m[s].model_fields["parameters"].default_factory
              assert model_params_type in set_params_input, f"Stage {s} set_parameters input type mismatch"

            else:
              set_params_input = all_s[s].set_parameters.__annotations__["parameters"].__name__
              model_params_type = all_m[s].model_fields["parameters"].default_factory.__name__
              assert set_params_input == model_params_type, f"Stage {s} set_parameters input type mismatch"

        except AssertionError as e:
            print(e)
            failed = True
            
        for field, value in type(all_m[s].model_fields["parameters"].default_factory()).model_fields.items():
            try:
                if get_origin(value.annotation) is list:
                    item_type = get_args(value.annotation)[0]
                    meta = get_args(item_type)[1].metadata
                elif get_origin(value.annotation) is Literal:
                  continue  # Skip Literal types
                else:
                  meta = value.metadata
                min = [g.ge for g in meta if isinstance(g, annotated_types.Ge)] or [
                g.gt for g in meta if isinstance(g, annotated_types.Gt)]
                max = [g.le for g in meta if isinstance(g, annotated_types.Le)] or [
                g.lt for g in meta if isinstance(g, annotated_types.Lt)]
                assert min, f"Minimum not defined for field {field} in {s}"
                if field not in ["position", "delay"]:
                  assert max, f"Maximum not defined for field {field} in {s}"
                  assert min[0] < max[0], f"Range not correct for field {field} in {s}"
            except AssertionError as e:
                print(e)
                failed = True
                continue

    assert not failed, "Some stages failed the test."



if __name__ == "__main__":
  test_shared_edge_from_graph_input()