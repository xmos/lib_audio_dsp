"""Test limiter pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline


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
                    "config": {},
                    "parameters": {
                        "threshold_db": -6.0,
                        "attack_t": 0.01,
                        "release_t": 0.1
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoRMSLimiter",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "audio_in",
                    "output": [0, 1]
                }
            ],
            "outputs": [
                {
                    "name": "audio_out",
                    "input": [2, 3]
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
                    "config": {},
                    "parameters": {
                        "threshold_db": -3.0,
                        "attack_t": 0.005,
                        "release_t": 0.05
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoPeakLimiter",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "audio_in",
                    "output": [0, 1]
                }
            ],
            "outputs": [
                {
                    "name": "audio_out",
                    "input": [2, 3]
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
                    "config": {},
                    "parameters": {
                        "threshold_db": -1.0,
                        "attack_t": 0.001,
                        "release_t": 0.02
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoHardPeakLimiter",
                        "thread": 0
                    }
                }
            ],
            "inputs": [
                {
                    "name": "audio_in",
                    "output": [0, 1]
                }
            ],
            "outputs": [
                {
                    "name": "audio_out",
                    "input": [2, 3]
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



if __name__ == "__main__":
    test_rms_limiter_pipeline()
    print("\n" + "="*50 + "\n")
    test_peak_limiter_pipeline()
    print("\n" + "="*50 + "\n")
    test_hard_peak_limiter_pipeline() 