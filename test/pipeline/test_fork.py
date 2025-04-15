# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages.signal_chain.Fork
"""
import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages.signal_chain import Fork
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np
from filelock import FileLock
import os
import shutil

PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

def do_test(p, in_ch, out_ch, folder_name):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare
    """
    app_dir = PKG_DIR / folder_name
    os.makedirs(app_dir, exist_ok=True)
    infile = app_dir / "instage.wav"
    outfile = app_dir / "outstage.wav"
    n_samps, rate = 1024, 48000

    with FileLock(build_utils.PIPELINE_BUILD_LOCK):
        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline_default")
        target = "default"
        # Build pipeline test executable. This will download xscope_fileio if not present
        build_utils.build(APP_DIR, BUILD_DIR, target)
        os.makedirs(app_dir / "bin", exist_ok=True)
        shutil.copytree(APP_DIR / "bin", app_dir / "bin", dirs_exist_ok=True)

    sig0 = np.linspace(-2**26, 2**26, n_samps, dtype=np.int32)  << 4 # numbers which should be unmodified through pipeline
                                                                     # data formats
    sig1 = np.linspace(2**26, -2**26, n_samps, dtype=np.int32)  << 4
    if len(in_ch) == 2:
        sig = np.stack((sig0, sig1), axis=1)
    else:
        sig = sig0.reshape((n_samps, 1))
    audio_helpers.write_wav(infile, rate, sig)

    xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, 2, 1)

    _, out_data = audio_helpers.read_wav(outfile)
    sim_out = p.executor().process(sig).data
    for in_i, out_i in zip(in_ch, out_ch):
        np.testing.assert_equal(sig[:, in_i], out_data[:, out_i])
        np.testing.assert_equal(sig[:, in_i], sim_out[:, out_i])

@pytest.mark.group0
@pytest.mark.parametrize("inputs, fork_output", [(2, 0),
                                                 (2, 1),
                                                 (1, 0)])
def test_fork(fork_output, inputs):
    """
    Basic check that the for stage correctly copies data to the expected outputs.
    """
    channels = inputs
    p, i = Pipeline.begin(channels)
    count = 2
    fork = p.stage(Fork, i, count = count)
    assert len(fork.forks) == count
    for f in fork.forks:
        assert len(f) == channels

    p.set_outputs(fork.forks[fork_output])

    if inputs == 1:
        do_test(p, [0], (0, 1), folder_name=f"fork_{inputs}_{fork_output}")
    else:
        do_test(p, (0, 1), (0, 1), folder_name=f"fork_{inputs}_{fork_output}")

@pytest.mark.group0
def test_fork_copies():
    """
    Check we can duplicate a channel
    """
    channels = 2
    p, i = Pipeline.begin(channels, frame_size=2)
    fork = p.stage(Fork, i, count = 2)
    p.set_outputs(fork.forks[0][0] + fork.forks[1][0])

    # input channel 0 comes out both outputs
    do_test(p, (0, 0), (0, 1), folder_name=f"fork_copy")
