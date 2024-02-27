# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages.signal_chain.Fork
"""
import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages.signal_chain import Adder
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np


PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

def do_test(p, in_ch, out_ch):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare
    """
    infile = "inadder.wav"
    outfile = "outadder.wav"
    n_samps, rate = 10, 48000

    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    sig0 = np.linspace(-2**26, 2**26, n_samps, dtype=np.int32)  << 4 # numbers which should be unmodified through pipeline
                                                                     # data formats
    sig1 = np.linspace(2**26, -2**26, n_samps, dtype=np.int32)  << 4
    sig = np.stack((sig0, sig1), axis=1)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, 2, 1)

    _, out_data = audio_helpers.read_wav(outfile)
    for in_i, out_i in zip(in_ch, out_ch):
        np.testing.assert_equal(sig[:, in_i], out_data[:, out_i])

@pytest.mark.parametrize("fork_output", (0, 1))
def test_adder(fork_output):
    """
    Basic check that the for stage correctly copies data to the expected outputs.
    """
    channels = 2
    p = Pipeline(channels)
    with p.add_thread() as t:
        adder = t.stage(Adder, p.i)
    p.set_outputs(adder.o)

    do_test(p, (0, 1), (0, 1))

if __name__ == "__main__":
    test_adder(0)
