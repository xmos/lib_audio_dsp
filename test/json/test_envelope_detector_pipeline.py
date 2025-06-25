# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Test envelope detector pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline, pipeline_to_dspjson


def test_peak_envelope_detector_pipeline():
    """Test creating a peak envelope detector pipeline."""
    print("Creating peak envelope detector pipeline...")
    
    # Create a peak envelope detector pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_envelope_detector",
        "producer_version": "0.1",
        "graph": {
            "name": "Peak Envelope Detector",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "EnvelopeDetectorPeak",
                    "parameters": {
                        "attack_t": 0.01,
                        "release_t": 0.1
                    },
                    "placement": {
                        "input": [["inputs", 0]],
                        "output": [],
                        "name": "PeakDetector",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "inputs",
                    "output": [["inputs", 0], ["inputs", 1]]
                }
            ],
            "outputs": [
                {
                    "name": "outputs",
                    "input": [["inputs", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)

    # Find our envelope detector stage
    detector_stage = None
    for stage in pipeline.stages:
        if stage.name == "envelope_detector_peak":
            detector_stage = stage
            break
            
    assert detector_stage is not None, "Could not find Peak Envelope Detector stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    

def test_rms_envelope_detector_pipeline():
    """Test creating an RMS envelope detector pipeline."""
    print("Creating RMS envelope detector pipeline...")
    
    # Create an RMS envelope detector pipeline JSON
    pipeline_json = {
        "ir_version": 1,
        "producer_name": "test_envelope_detector",
        "producer_version": "0.1",
        "graph": {
            "name": "RMS Envelope Detector",
            "fs": 48000,
            "nodes": [
                {
                    "op_type": "EnvelopeDetectorRMS",
                    "parameters": {
                        "attack_t": 0.05,
                        "release_t": 0.2
                    },
                    "placement": {
                        "input": [["inputs", 0]],
                        "output": [],
                        "name": "RMSDetector",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "inputs",
                    "output": [["inputs", 0], ["inputs", 1]]
                }
            ],
            "outputs": [
                {
                    "name": "outputs",
                    "input": [["inputs", 1]]
                }
            ]
        }
    }
    
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    # Find our envelope detector stage
    detector_stage = None
    for stage in pipeline.stages:
        if stage.name == "envelope_detector_rms":
            detector_stage = stage
            break
            
    assert detector_stage is not None, "Could not find RMS Envelope Detector stage in pipeline"

    new_json = pipeline_to_dspjson(pipeline)
    assert dsp_json.graph == new_json.graph, "Pipeline JSON does not match original"
    

if __name__ == "__main__":
    test_peak_envelope_detector_pipeline()
    print("\n" + "="*50 + "\n")
    test_rms_envelope_detector_pipeline()