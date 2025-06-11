# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test reverb pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_plate_reverb_pipeline():
    """Test creating a plate reverb pipeline."""
    print("Creating plate reverb pipeline...")
    
    # Create a plate reverb pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_reverb",
        "producer_version": "0.1",
        "graph": {
            "name": "Plate Reverb",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "ReverbPlateStereo",
                    "config": {
                        "max_predelay": 30.0
                    },
                    "parameters": {
                        "predelay": 15.0,
                        "width": 0.8,
                        "damping": 0.4,
                        "decay": 0.6,
                        "early_diffusion": 0.7,
                        "late_diffusion": 0.6,
                        "bandwidth": 0.9,
                        "wet_dry_mix": 0.4
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoPlateReverb",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "inputs",
                    "output": [0, 1]
                }
            ],
            "outputs": [
                {
                    "name": "outputs",
                    "input": [2, 3]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our reverb stage
    reverb_stage = None
    for stage in pipeline.stages:
        if stage.name == "reverb_plate":
            reverb_stage = stage
            break
            
    assert reverb_stage is not None, "Could not find Plate Reverb stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    

if __name__ == "__main__":
    test_plate_reverb_pipeline()