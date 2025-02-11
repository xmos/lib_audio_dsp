"""Test FIR filter pipeline creation."""

from audio_dsp.design.parse_json import DspJson, make_pipeline
import numpy as np
import tempfile
import os


def test_fir_pipeline():
    """Test creating an FIR filter pipeline."""
    print("Creating FIR filter pipeline...")
    
    # First create a temporary file with FIR coefficients
    coeffs = np.array([1.0, 0.5, 0.25, 0.125])  # Simple lowpass filter coefficients
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        np.savetxt(f.name, coeffs)
        coeffs_path = f.name
    
    try:
        # Create an FIR filter pipeline JSON
        pipeline_json = {
            "ir_version": 1,
            "producer_name": "test_fir",
            "producer_version": "0.1",
            "graph": {
                "name": "FIR Filter",
                "fs": 48000,
                "nodes": [
                    {
                        "op_type": "FirDirect",
                        "config": {
                            "coeffs_path": coeffs_path
                        },
                        "parameters": {},
                        "placement": {
                            "input": [0],
                            "output": [1],
                            "name": "MonoFIR",
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
                        "input": [1]
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
        # Find our FIR stage
        fir_stage = None
        for stage in pipeline.stages:
            if stage.name == "fir_direct":
                fir_stage = stage
                break
                
        assert fir_stage is not None, "Could not find FIR stage in pipeline"
        print("✓ Found FIR stage in pipeline")
        
        print("\nAll tests passed successfully! 🎉")
        
    finally:
        # Clean up the temporary coefficients file
        os.unlink(coeffs_path)


if __name__ == "__main__":
    test_fir_pipeline() 