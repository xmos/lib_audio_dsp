"""Test envelope detector pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline


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
                    "config": {},
                    "parameters": {
                        "attack_t": 0.01,
                        "release_t": 0.1
                    },
                    "placement": {
                        "input": [0],
                        "output": [],
                        "name": "PeakDetector",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "audio_in",
                    "output": [0]
                }
            ],
            "outputs": [
                {
                    "name": "audio_out",
                    "input": [0]
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
                    "config": {},
                    "parameters": {
                        "attack_t": 0.05,
                        "release_t": 0.2
                    },
                    "placement": {
                        "input": [0],
                        "output": [],
                        "name": "RMSDetector",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "audio_in",
                    "output": [0]
                }
            ],
            "outputs": [
                {
                    "name": "audio_out",
                    "input": [0]
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


if __name__ == "__main__":
    test_peak_envelope_detector_pipeline()
    print("\n" + "="*50 + "\n")
    test_rms_envelope_detector_pipeline() 