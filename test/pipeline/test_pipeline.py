# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path
import scipy.io.wavfile
import numpy as np
import subprocess
import pytest
from copy import deepcopy

from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages.biquad import Biquad
from audio_dsp.stages.cascaded_biquads import CascadedBiquads
from audio_dsp.stages.signal_chain import Bypass
from audio_dsp.stages.limiter import LimiterRMS, LimiterPeak
from audio_dsp.design.pipeline import generate_dsp_main
import audio_dsp.dsp.signal_gen as gen

from python import build_utils, run_pipeline_xcoreai, audio_helpers
from stages.add_n import AddN

# Test config
PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"
num_in_channels = 2
num_out_channels = 2
input_dtype = np.int32
test_duration = 0.1 # in seconds
infile = "test_input.wav"
outfile = "test_output.wav"
Fs = 48000

def create_pipeline():
    # Create pipeline
    p = Pipeline(num_in_channels)

    with p.add_thread() as t:
        bi = t.stage(Biquad, p.i)
        cb = t.stage(CascadedBiquads, bi.o)
        by = t.stage(Bypass, cb.o)
        lp = t.stage(LimiterPeak, by.o)
        lr = t.stage(LimiterRMS, lp.o)
    with p.add_thread() as t:                # ch   0   1
        by1 = t.stage(Bypass, lr.o)

    p.set_outputs(by1.o)
    stages = 2
    return p, stages

def test_pipeline():
    """
    Basic test playing a sine wave through a stage
    """
    p, n_stages = create_pipeline()

    # Autogenerate C code
    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"

    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    outfile_py = Path(outfile).parent / (str(Path(outfile).stem) + '_py.wav')
    outfile_c = Path(outfile).parent / (str(Path(outfile).stem) + '_c.wav')

    # Generate input
    input_sig_py = np.empty((int(Fs*test_duration), num_in_channels), dtype=np.float64)
    for i in range(num_in_channels):
        input_sig_py[:, i] = (gen.sin(fs=Fs, length=test_duration, freq=1000, amplitude=0.1, precision=27)).T

    if (input_dtype == np.int32) or (input_dtype == np.int16):
        input_sig_c = np.array(input_sig_py * np.iinfo(input_dtype).max, dtype=input_dtype)
    else:
        input_sig_c = deepcopy(input_sig_py)

    scipy.io.wavfile.write(infile, Fs, input_sig_c)

    # Run python
    sim_sig = p.executor().process(input_sig_py)
    np.testing.assert_equal(sim_sig, input_sig_py)
    sim_sig = np.array(sim_sig * np.iinfo(np.int32).max, dtype=np.int32)
    audio_helpers.write_wav(outfile_py, Fs, sim_sig)

    # Run C
    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile_c, num_out_channels, n_stages)

    # pipeline operates at q27 so truncate the input to match expected output
    exp_fs, exp_sig = audio_helpers.read_and_truncate(infile)

    # Compare
    out_fs, out_sig_py = audio_helpers.read_and_truncate(outfile_py)
    out_fs, out_sig_c = audio_helpers.read_and_truncate(outfile_c) # For C this should do nothing since the Q27 -> Q31 at the output ensures that the lower 4 bits are 0
    np.testing.assert_equal(out_sig_py, exp_sig[:out_sig_py.shape[0],:]) # Compare python with input
    np.testing.assert_equal(out_sig_c, exp_sig[:out_sig_c.shape[0],:]) # Compare C with input


INT32_MIN = -(2**31)
INT32_MAX = (-INT32_MIN) - 1
@pytest.mark.parametrize("input,add,output", [(3, 3, 3 << 4),  # input too insignificant, output is shifted
                                              (-3, 0, -1 << 4), # negative small numbers truncate in the negative direction
                                              (INT32_MIN, -3, INT32_MIN),
                                              (INT32_MAX, 3, INT32_MAX),
                                              (INT32_MAX, -3, ((INT32_MAX >>4) - 3) << 4)])
def test_pipeline_q27(input, add, output):
    """
    Check that the pipeline operates at q5.27 and outputs saturated Q1.31

    This is done by having a stage which adds a constant to its input. Inputing
    small values will result in the small value being too insgnificant and shifted out.
    and the result is the constant shifted up.

    Check for saturation by adding a constant to large values and checking the output hasn't
    overflowed
    """
    infile = "inq27.wav"
    outfile = "outq27.wav"
    n_samps, channels, rate = 1024, 2, 48000

    p = Pipeline(channels)
    with p.add_thread() as t:
        addn = t.stage(AddN, p.i, n=add)
    p.set_outputs(addn.o)

    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    sig = np.multiply(np.ones((n_samps, channels), dtype=np.int32), input, dtype=np.int32)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, num_out_channels, pipeline_stages=1)

    expected = np.multiply(np.ones((n_samps, channels), dtype=np.int32), output, dtype=np.int32)
    _, out_data = audio_helpers.read_wav(outfile)
    np.testing.assert_equal(expected, out_data)

def test_complex_pipeline():
    """
    Generate a multithreaded pipeline and check the output is as expected
    """
    infile = "incomplex.wav"
    outfile = "outcomplex.wav"
    n_samps, channels, rate = 1024, 2, 48000

    p = Pipeline(channels)
    with p.add_thread() as t:                # ch   0   1
        # this thread has both channels
        a = t.stage(AddN, p.i, n=1)          #     +1  +1
        a = t.stage(AddN, a.o, n=1)          #     +2  +2
        a0 = t.stage(AddN, a.o[:1], n=1)     #     +3  +2
    with p.add_thread() as t:
        # this thread has channel 1
        a1 = t.stage(AddN, a.o[1:], n=1)     #     +3  +3
        a1 = t.stage(AddN, a1.o, n=1)        #     +3  +4
        a1 = t.stage(AddN, a1.o, n=1)        #     +3  +5
    with p.add_thread() as t:
        # this thread has channel 0
        a0 = t.stage(AddN, a0.o, n=1)        #     +4  +5
    with p.add_thread() as t:
        # this thread has both channels
        a = t.stage(AddN, a0.o + a1.o, n=1)  #     +5  +6

    p.set_outputs(a.o)
    n_stages = 3  # 2 of the 4 threads are parallel

    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    in_val = 1000
    # expected output is +5 on left, +6 on right, in the Q1.27 format
    expected = (np.array([[5, 6]]*n_samps) + (in_val >> 4)) << 4
    sig = np.multiply(np.ones((n_samps, channels), dtype=np.int32), in_val, dtype=np.int32)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, num_out_channels, n_stages)
    _, out_data = audio_helpers.read_wav(outfile)
    np.testing.assert_equal(expected, out_data)

