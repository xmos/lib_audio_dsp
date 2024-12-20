# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.dsp.generic import Q_SIG
from stages.frame_count import FrameCount
from python import audio_helpers, build_utils, run_pipeline_xcoreai
import numpy as np
from pathlib import Path
from itertools import cycle
import pytest
import os
import shutil
from filelock import FileLock


PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

def shift_to_pipeline(i):
    return i << (31 - Q_SIG)

@pytest.mark.parametrize("frame_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_frame_size(frame_size):
    """
    Using a custom test stage that fills each frame of output with the index
    of that outputs position in the frame to check that changing the frame size
    in the pipeline design actually changes the frame size in the generated code.
    """
    p, i = Pipeline.begin(1, frame_size=frame_size)

    s = p.stage(FrameCount, i)
    p.set_outputs(s)
    
    app_dir = PKG_DIR / f"test_frame_size_{frame_size}"
    os.makedirs(app_dir, exist_ok=True)
    infile = app_dir / "inframe.wav"
    outfile = app_dir / "outframe.wav"
    n_samps, rate = 2048, 48000



    with FileLock("test_pipeline_build.lock"):
        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline_initialized")
        target = "default"
        # Build pipeline test executable. This will download xscope_fileio if not present
        build_utils.build(APP_DIR, BUILD_DIR, target)
        os.makedirs(app_dir / "bin", exist_ok=True)
        shutil.copytree(APP_DIR / "bin", app_dir / "bin", dirs_exist_ok=True)

    sig = np.zeros((n_samps, 1), dtype=np.int32)
    audio_helpers.write_wav(infile, rate, sig)

    xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, 1, 1)

    _, out_data = audio_helpers.read_wav(outfile)

    expected=cycle(range(frame_size))
    i = 0
    for i, (actual, expected) in enumerate(zip(out_data, expected)):
        assert actual == shift_to_pipeline(expected)
    assert i + 1 == n_samps
