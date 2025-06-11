# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test compressor pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_simple_compressor_pipeline():
    """Test creating a simple compressor pipeline."""
    print("Creating simple stereo compressor pipeline...")
    
    # Create a simple stereo compressor pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_compressor",
        "producer_version": "0.1",
        "graph": {
            "name": "Simple Compressor",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "CompressorRMS",
                    "parameters": {
                        "ratio": 4.0,
                        "threshold_db": -20.0,
                        "attack_t": 0.01,
                        "release_t": 0.2
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoCompressor",
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

    # Find our compressor stage
    compressor_stage = None
    for stage in pipeline.stages:
        if stage.name == "compressor_rms":
            compressor_stage = stage
            break
            
    assert compressor_stage is not None, "Could not find Compressor stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"


if __name__ == "__main__":
    test_simple_compressor_pipeline() 