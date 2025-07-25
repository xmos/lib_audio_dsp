# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test noise gate pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_noise_gate_pipeline():
    """Test creating a noise gate pipeline."""
    print("Creating noise gate pipeline...")
    
    # Create a noise gate pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_noise_gate",
        "producer_version": "0.1",
        "graph": {
            "name": "Noise Gate",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "NoiseGate",
                    "parameters": {
                        "threshold_db": -40.0,
                        "attack_t": 0.01,
                        "release_t": 0.1
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoNoiseGate",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "inputs",
                    "channels": 2
                }
            ],
            "outputs": [
                {
                    "name": "outputs",
                    "input": [["StereoNoiseGate", 0], ["StereoNoiseGate", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our noise gate stage
    gate_stage = None
    for stage in pipeline.stages:
        if stage.name == "noise_gate":  # This matches the config file name
            gate_stage = stage
            break
            
    assert gate_stage is not None, "Could not find Noise Gate stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    

if __name__ == "__main__":
    test_noise_gate_pipeline()