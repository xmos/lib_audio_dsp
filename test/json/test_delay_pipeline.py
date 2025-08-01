# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test delay pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson
from audio_dsp.stages.signal_chain import Delay


def test_simple_delay_pipeline():
    """Test creating a simple delay pipeline."""
    print("Creating simple stereo delay pipeline...")
    
    # Create a simple stereo delay pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_delay",
        "producer_version": "0.1",
        "graph": {
            "name": "Simple Delay",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "Delay",
                    "config": {
                        "max_delay": 2048,
                        "units": "samples"
                    },
                    "parameters": {
                        "delay": 1024,
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoDelay",
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
                    "input": [["StereoDelay", 0], ["StereoDelay", 1]],
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)

    # Find our delay stage
    delay_stage = None
    for stage in pipeline.stages:
        if isinstance(stage, Delay):
            delay_stage = stage
            break
            
    assert delay_stage is not None, "Could not find Delay stage in pipeline"
    assert delay_stage.max_delay == 2048, f"Expected max_delay 2048, got {delay_stage.max_delay}"
    assert delay_stage.units == "samples", f"Expected units 'samples', got {delay_stage.units}"
    assert delay_stage.parameters.delay == 1024, f"Expected delay 1024, got {delay_stage.parameters.delay}"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    


if __name__ == "__main__":
    test_simple_delay_pipeline() 