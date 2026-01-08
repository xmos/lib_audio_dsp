# Copyright 2025-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test noise suppressor expander pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_noise_suppressor_expander_pipeline():
    """Test creating a noise suppressor expander pipeline."""
    print("Creating noise suppressor expander pipeline...")
    
    # Create a noise suppressor expander pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_noise_suppressor",
        "producer_version": "0.1",
        "graph": {
            "name": "Noise Suppressor Expander",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "NoiseSuppressorExpander",
                    "parameters": {
                        "ratio": 3.0,
                        "threshold_db": -35.0,
                        "attack_t": 0.005,
                        "release_t": 0.120
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoNoiseSuppressor",
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
                    "input": [["StereoNoiseSuppressor", 0], ["StereoNoiseSuppressor", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our noise suppressor stage
    suppressor_stage = None
    for stage in pipeline.stages:
        if stage.name == "noise_suppressor_expander":
            suppressor_stage = stage
            break
            
    assert suppressor_stage is not None, "Could not find Noise Suppressor Expander stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    

if __name__ == "__main__":
    test_noise_suppressor_expander_pipeline()