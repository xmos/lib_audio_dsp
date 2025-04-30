"""Test reverb pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline


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
                        "predelay": 15.0,
                        "max_predelay": 30.0
                    },
                    "parameters": {
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
    
    print("Parsing JSON and creating pipeline...")
    dsp_json = DspJson(**pipeline_json)
    pipeline = make_pipeline(dsp_json)
    
    print("\nStages in pipeline:")
    for i, stage in enumerate(pipeline.stages):
        print(f"Stage {i}: {stage.name} (type: {type(stage).__name__})")
    
    print("\nValidating pipeline...")
    # Find our reverb stage
    reverb_stage = None
    for stage in pipeline.stages:
        if stage.name == "reverb_plate":
            reverb_stage = stage
            break
            
    assert reverb_stage is not None, "Could not find Plate Reverb stage in pipeline"
    print("✓ Found Plate Reverb stage in pipeline")
    
    print("\nAll tests passed successfully! 🎉")


if __name__ == "__main__":
    test_plate_reverb_pipeline()