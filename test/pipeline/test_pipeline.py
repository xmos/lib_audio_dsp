
from pathlib import Path
import scipy.io.wavfile
import numpy as np
import subprocess
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages.biquad import Biquad
from audio_dsp.stages.cascaded_biquads import CascadedBiquads
from audio_dsp.design.pipeline import generate_dsp_main
import sys
sys.path.append("python")
import build_utils
import run_pipeline_xcoreai
import audio_helpers

# Test config
PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"
num_in_channels = 2
num_out_channels = 2
input_dtype = np.int32
infile = "test_input.wav"
outfile = "test_output.wav"
compare = True


def create_pipeline():
    # Create pipeline
    p = Pipeline(num_in_channels)

    with p.add_thread() as t:
        butter = t.stage(CascadedBiquads, p.i[:num_in_channels])

    p.set_outputs(butter.o)
    return p


def test_pipeline():
    p = create_pipeline()

    # Autogenerate C code
    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"

    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    # Generate input
    audio_helpers.generate_test_signal(infile, type="sine", fs=48000, duration=10, amplitude=0.8, num_channels=2, sig_dtype=input_dtype)

    # Run
    # TODO Run from a tmp directory
    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, num_out_channels)

    # Compare
    if compare:
        all_close, maxdiff, delay = audio_helpers.correlate_and_diff(outfile, infile, [0,num_out_channels-1], [0,num_out_channels-1], 0, 0, 1e-8)
        assert maxdiff == 0, "Pipline input and output not bit-exact"
        assert all_close == True, "Pipline input and output not close enough"


if __name__ == "__main__":
    test_pipeline()
