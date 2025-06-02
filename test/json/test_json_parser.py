from pathlib import Path
import annotated_types

from audio_dsp.design.parse_json import Graph, make_pipeline, insert_forks, DspJson

from audio_dsp.models.stage import all_models
from audio_dsp.stages import all_stages

def test_no_shared_edge():
    json_str = """
    {
      "name": "No shared edge",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "VolumeControl_1",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [1]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_no_shared_edge"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def test_shared_edge_from_graph_input():
    json_str = """
    {
      "name": "Shared edge from graph input",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "VolumeControl_A",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [0],
            "output": [2],
            "name": "Mixer_B",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [1]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_shared_edge_from_graph_input"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def test_shared_edge_from_producer_node():
    json_str = """
    {
      "name": "Shared edge from producer node",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "Producer",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1],
            "output": [2],
            "name": "Consumer_1",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1],
            "output": [3],
            "name": "Consumer_2",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [2]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_shared_edge_from_producer_node"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def test_shared_edge_with_graph_output():
    json_str = """
    {
      "name": "Shared edge with graph output",
      "fs": 44100,
      "inputs": [{
          "name": "audio_in",
          "output": [0]
      }],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [1],
            "name": "Producer",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1],
            "output": [2],
            "name": "Consumer",
            "thread": 0
          }
        }
      ],
      "outputs": [{
          "name": "audio_out",
          "input": [1]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_shared_edge_with_graph_output"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


def test_again():
    json_str = """
    {
      "name": "Stereo Mixer with Volume",
      "fs": 48000,
      "inputs": [{
          "name": "stereo_in",
          "output": [0, 1]
      }],
      "nodes": [
        {"op_type": "Mixer", "placement": {"input": [0, 1], "name": "Mixer", "output": [2], "thread": 0}},
        {"op_type": "VolumeControl", "placement": {"input": [2], "name": "Volume", "output": [3], "thread": 0}}
      ],
      "outputs": [{
          "name": "stereo_out",
          "input": [3, 3]
      }]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_again"))
    new_graph = insert_forks(graph)
    print(f"Before insert_forks: {graph.model_dump_json()}")
    print(f"After insert_forks: {new_graph.model_dump_json()}")


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
        {"name": "mono_in", "output": [0]},
        {"name": "stereo_in", "output": [1, 2]}
      ],
      "nodes": [
        {
          "op_type": "VolumeControl",
          "config": {},
          "placement": {
            "input": [0],
            "output": [3],
            "name": "VolumeControl_A",
            "thread": 0
          }
        },
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [1, 2],
            "output": [4],
            "name": "Mixer_B",
            "thread": 0
          }
        }
      ],
      "outputs": [
        {"name": "mono_out", "input": [3]},
        {"name": "stereo_out", "input": [4, 3]}
      ]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_multiple_inputs_outputs_non_shared"))
    new_graph = insert_forks(graph)
    print("Test: Multiple Inputs and Outputs Non-Shared Test")
    print(f"Before insert_forks: {graph.model_dump_json(indent=2)}")
    print(f"After insert_forks: {new_graph.model_dump_json(indent=2)}")


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
        {"name": "input1", "output": [0]},
        {"name": "input2", "output": [1]}
      ],
      "nodes": [
        {
          "op_type": "Mixer",
          "config": {},
          "placement": {
            "input": [0, 1],
            "output": [2],
            "name": "Mixer",
            "thread": 0
          }
        }
      ],
      "outputs": [
        {"name": "output1", "input": [2]},
        {"name": "output2", "input": [2]}
      ]
    }
    """
    graph = Graph.model_validate_json(json_str)
    a = make_pipeline(
        DspJson(ir_version=1, producer_name="test", producer_version="1.0", graph=graph)
    )
    a.draw(Path("test_multiple_inputs_outputs_shared"))
    new_graph = insert_forks(graph)
    print("Test: Multiple Inputs and Outputs Shared Test")
    print(f"Before insert_forks: {graph.model_dump_json(indent=2)}")
    print(f"After insert_forks: {new_graph.model_dump_json(indent=2)}")


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
            set_params_input = all_s[s].set_parameters.__annotations__["parameters"].__name__
            model_params_type = all_m[s].model_fields["parameters"].default_factory.__name__
            assert set_params_input == model_params_type, f"Stage {s} set_parameters input type mismatch"
        except AssertionError as e:
            print(e)
            failed = True
            
        for field, value in all_m[s].model_fields["parameters"].default_factory().model_fields.items():
            try:
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
  test_all_stages_models()