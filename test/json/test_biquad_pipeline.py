"""Test biquad filter pipeline creation.

# Instructions for Fixing a Stage Implementation

## 1. Check Related Files
First, locate and check these files for the stage you're fixing:
- Stage implementation: `python/audio_dsp/stages/{stage_name}.py`
- Stage config: `stage_config/{stage_name}.yaml`
- Parameter types: `python/audio_dsp/models/{stage_name}_model.py`

## 2. Align Parameters with YAML
1. Compare the yaml config with your stage's parameters
2. Split parameters into:
   - Config values: compile-time settings (e.g. max_delay)
   - Runtime parameters: values that can change during operation
3. Make sure parameter names and types match exactly with the yaml
4. Use pydantic models for parameter validation

## 3. Fix Stage Implementation
1. Store config values as stage attributes (e.g. self.max_delay)
2. Store runtime parameters in self.parameters
3. Implement `__init__` to handle both:
   - Direct parameter initialization
   - Dictionary-based initialization from config/parameters
4. Implement `set_parameters` method for runtime updates
5. Add proper pipeline integration:
   - Handle inputs/outputs correctly
   - Set up proper placement information
   - Handle thread assignment

## 4. Test Pipeline Integration
1. Create a test in `python/audio_dsp/tests/test_{stage_name}_pipeline.py`
2. Test both stereo and mono configurations
3. Test parameter updates
4. Verify stage can be found and configured in pipeline

## 5. Common Gotchas
- Match parameter names exactly with yaml
- Handle default values carefully
- Ensure proper input/output edge handling
- Don't forget to register stage in `__init__.py`

## 6. Validation Steps
1. Compare with yaml: `stage_config/{stage_name}.yaml`
2. Verify pipeline integration: `design/pipeline.py`
3. Test parameter updates
4. Verify thread handling
5. Check edge handling
"""

from audio_dsp.design.parse_json import DspJson, make_pipeline
from audio_dsp.models.biquad_model import Biquad


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
                    "config": {},
                    "parameters": {
                        "filter_type": {
                            "type": "lowpass",
                            "filter_freq": 1000,
                            "q_factor": 0.707
                        },
                        "slew_rate": 0.5
                    },
                    "placement": {
                        "input": [0, 1],
                        "output": [2, 3],
                        "name": "StereoBiquad",
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
    # Find our biquad stage
    biquad_stage = None
    for stage in pipeline.stages:
        if stage.name == "biquad":
            biquad_stage = stage
            break
            
    assert biquad_stage is not None, "Could not find Biquad stage in pipeline"
    print("✓ Found Biquad stage in pipeline")
    
    assert biquad_stage.parameters.filter_type.type == "lowpass", \
        f"Expected filter_type 'lowpass', got {biquad_stage.parameters.filter_type.type}"
    print("✓ filter_type is correct")
    
    assert biquad_stage.parameters.filter_type.filter_freq == 1000, \
        f"Expected filter_freq 1000, got {biquad_stage.parameters.filter_type.filter_freq}"
    print("✓ filter_freq is correct")
    
    assert biquad_stage.parameters.filter_type.q_factor == 0.707, \
        f"Expected q_factor 0.707, got {biquad_stage.parameters.filter_type.q_factor}"
    print("✓ q_factor is correct")
    
    assert biquad_stage.parameters.slew_rate == 0.5, \
        f"Expected slew_rate 0.5, got {biquad_stage.parameters.slew_rate}"
    print("✓ slew_rate is correct")
    
    print("\nAll tests passed successfully! 🎉")


if __name__ == "__main__":
    test_simple_biquad_pipeline() 