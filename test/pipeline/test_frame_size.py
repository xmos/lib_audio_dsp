from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.dsp.generic import Q_SIG
from stages.frame_count import FrameCount
from python import audio_helpers, build_utils, run_pipeline_xcoreai
import numpy as np
from pathlib import Path
from itertools import cycle
import pytest


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
    p = Pipeline(1, frame_size=frame_size)
    t = p.add_thread()

    s = t.stage(FrameCount, p.i)
    p.set_outputs(s.o)

    infile = "inframe.wav"
    outfile = "outframe.wav"
    n_samps, rate = 2048, 48000

    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    sig = np.zeros((n_samps, 1), dtype=np.int32)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, 1, 1)

    _, out_data = audio_helpers.read_wav(outfile)

    expected=cycle(range(frame_size))
    i = 0
    for i, (actual, expected) in enumerate(zip(out_data, expected)):
        assert actual == shift_to_pipeline(expected)
    assert i + 1 == n_samps

