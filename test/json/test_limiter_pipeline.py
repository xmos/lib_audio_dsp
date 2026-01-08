# Copyright 2025-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test limiter pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_rms_limiter_pipeline():
    """Test creating an RMS limiter pipeline."""
    print("Creating RMS limiter pipeline...")
    
    # Create an RMS limiter pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_limiter",
        "producer_version": "0.1",
        "graph": {
            "name": "RMS Limiter",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "LimiterRMS",
                    "parameters": {
                        "threshold_db": -6.0,
                        "attack_t": 0.01,
                        "release_t": 0.1
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoRMSLimiter",
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
                    "input": [["StereoRMSLimiter", 0], ["StereoRMSLimiter", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our limiter stage
    limiter_stage = None
    for stage in pipeline.stages:
        if stage.name == "limiter_rms":
            limiter_stage = stage
            break
            
    assert limiter_stage is not None, "Could not find RMS Limiter stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    


def test_peak_limiter_pipeline():
    """Test creating a peak limiter pipeline."""
    print("Creating peak limiter pipeline...")
    
    # Create a peak limiter pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_limiter",
        "producer_version": "0.1",
        "graph": {
            "name": "Peak Limiter",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "LimiterPeak",
                    "parameters": {
                        "threshold_db": -3.0,
                        "attack_t": 0.005,
                        "release_t": 0.05
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoPeakLimiter",
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
                    "input": [["StereoPeakLimiter", 0], ["StereoPeakLimiter", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)

    # Find our limiter stage
    limiter_stage = None
    for stage in pipeline.stages:
        if stage.name == "limiter_peak":
            limiter_stage = stage
            break
            
    assert limiter_stage is not None, "Could not find Peak Limiter stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    

def test_hard_peak_limiter_pipeline():
    """Test creating a hard peak limiter pipeline."""
    print("Creating hard peak limiter pipeline...")
    
    # Create a hard peak limiter pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_limiter",
        "producer_version": "0.1",
        "graph": {
            "name": "Hard Peak Limiter",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "HardLimiterPeak",
                    "parameters": {
                        "threshold_db": -1.0,
                        "attack_t": 0.001,
                        "release_t": 0.02
                    },
                    "placement": {
                        "input": [["inputs", 0], ["inputs", 1]],
                        "name": "StereoHardPeakLimiter",
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
                    "input": [["StereoHardPeakLimiter", 0], ["StereoHardPeakLimiter", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our limiter stage
    limiter_stage = None
    for stage in pipeline.stages:
        if stage.name == "hard_limiter_peak":
            limiter_stage = stage
            break
            
    assert limiter_stage is not None, "Could not find Hard Peak Limiter stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    


if __name__ == "__main__":
    test_rms_limiter_pipeline()
    print("\n" + "="*50 + "\n")
    test_peak_limiter_pipeline()
    print("\n" + "="*50 + "\n")
    test_hard_peak_limiter_pipeline()