# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test biquad filter pipeline creation.
"""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson
from audio_dsp.models.biquad import Biquad


def test_simple_biquad_pipeline():
    """Test creating a simple biquad filter pipeline."""
    print("Creating simple stereo biquad pipeline...")
    
    # Create a simple stereo biquad pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_biquad",
        "producer_version": "0.1",
        "graph": {
            "name": "Simple Biquad",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "Biquad",
                    "parameters": {
                        "filter_type": {
                            "type": "lowpass",
                            "filter_freq": 1000,
                            "q_factor": 0.707
                        },
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoBiquad",
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
                    "input": [["StereoBiquad", 0], ["StereoBiquad", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our biquad stage
    biquad_stage = None
    for stage in pipeline.stages:
        if stage.name == "biquad":
            biquad_stage = stage
            break
            
    assert biquad_stage is not None, "Could not find Biquad stage in pipeline"
    
    assert biquad_stage.parameters.filter_type.type == "lowpass", \
        f"Expected filter_type 'lowpass', got {biquad_stage.parameters.filter_type.type}"
    
    assert biquad_stage.parameters.filter_type.filter_freq == 1000, \
        f"Expected filter_freq 1000, got {biquad_stage.parameters.filter_type.filter_freq}"
    
    assert biquad_stage.parameters.filter_type.q_factor == 0.707, \
        f"Expected q_factor 0.707, got {biquad_stage.parameters.filter_type.q_factor}"
    
    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"

if __name__ == "__main__":
    test_simple_biquad_pipeline() 