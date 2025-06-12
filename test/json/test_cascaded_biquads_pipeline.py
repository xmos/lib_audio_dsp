# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test cascaded biquads pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_parametric_eq_pipeline():
    """Test creating a parametric EQ pipeline."""
    print("Creating parametric EQ pipeline...")
    
    # Create a parametric EQ pipeline JSON with different filter types
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_cascaded_biquads",
        "producer_version": "0.1",
        "graph": {
            "name": "Parametric EQ",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "ParametricEq8b",
                    "parameters": {
                        "filters": [
                            {
                                "type": "lowpass",
                                "filter_freq": 10000,
                                "q_factor": 0.707
                            },
                            {
                                "type": "highpass",
                                "filter_freq": 20,
                                "q_factor": 0.707
                            },
                            {
                                "type": "peaking",
                                "filter_freq": 1000,
                                "q_factor": 2.0,
                                "gain_db": 6.0
                            },
                            {
                                "type": "bypass"
                            },
                            {
                                "type": "bypass"
                            },
                            {
                                "type": "bypass"
                            },
                            {
                                "type": "bypass"
                            },
                            {
                                "type": "bypass"
                            }
                        ]
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoEQ",
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

    # Find our parametric EQ stage
    eq_stage = None
    for stage in pipeline.stages:
        if stage.name == "cascaded_biquads":
            eq_stage = stage
            break
            
    assert eq_stage is not None, "Could not find Parametric EQ stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"

if __name__ == "__main__":
    test_parametric_eq_pipeline() 