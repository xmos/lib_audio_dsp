# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path
import scipy.io.wavfile
import numpy as np
import subprocess
import pytest
from copy import deepcopy
import re
import subprocess
from filelock import FileLock
import os
import shutil

from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages.biquad import Biquad
from audio_dsp.stages.cascaded_biquads import CascadedBiquads
from audio_dsp.stages.signal_chain import Bypass
from audio_dsp.stages.limiter import LimiterRMS, LimiterPeak
from audio_dsp.design.pipeline import generate_dsp_main
import audio_dsp.dsp.signal_gen as gen
from audio_dsp.dsp.generic import HEADROOM_BITS
import audio_dsp.dsp.utils as utils

from python import build_utils, run_pipeline_xcoreai, audio_helpers
from stages.add_n import AddN
from python.run_pipeline_xcoreai import FORCE_ADAPTER_ID

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


def gen_build(app_dir, p, target):
    with FileLock(build_utils.PIPELINE_BUILD_LOCK):
        # Autogenerate C code
        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline_default")

        # Build pipeline test executable. This will download xscope_fileio if not present
        build_utils.build(APP_DIR, BUILD_DIR, target)
        os.makedirs(app_dir / "bin", exist_ok=True)
        shutil.copytree(APP_DIR / "bin", app_dir / "bin", dirs_exist_ok=True)

def create_pipeline():
    # Create pipeline
    p, i = Pipeline.begin(num_in_channels)

    bi = p.stage(Biquad, i, label="biquad")
    cb = p.stage(CascadedBiquads, bi, label="casc_biquad")
    by = p.stage(Bypass, cb, label="byp_1")
    lp = p.stage(LimiterPeak, by)
    lr = p.stage(LimiterRMS, lp)
    p.next_thread()                # ch   0   1
    by1 = p.stage(Bypass, lr, label="byp_2")

    p.set_outputs(by1)
    stages = 2
    return p, stages

def test_pipeline():
    """
    Basic test playing a sine wave through a stage
    """
    p, n_stages = create_pipeline()

    app_dir = PKG_DIR / "test_pipeline"
    os.makedirs(app_dir, exist_ok=True)
    target = "default"

    gen_build(app_dir, p, target)

    outfile_py = Path(outfile).parent / (str(Path(outfile).stem) + '_py.wav')
    outfile_c = Path(outfile).parent / (str(Path(outfile).stem) + '_c.wav')

    # Generate input
    input_sig_py = np.empty((int(Fs*test_duration), num_in_channels), dtype=np.float64)
    for i in range(num_in_channels):
        # precision of 28 gives Q27 signal
        input_sig_py[:, i] = (gen.sin(fs=Fs, length=test_duration, freq=1000, amplitude=0.1, precision=28)).T

    if (input_dtype == np.int32) or (input_dtype == np.int16):
        int_sig = (np.array(input_sig_py) * utils.Q_max(27)).astype(np.int32)
        if input_dtype == np.int32:
            qformat = 31
            int_sig <<= 4
        else:
            qformat = 15
            int_sig >>= (27-15)

        input_sig_c = np.clip(int_sig, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
    else:
        input_sig_c = deepcopy(input_sig_py)

    print(input_sig_py)
    print(input_sig_c)

    scipy.io.wavfile.write(infile, Fs, input_sig_c)

    # Run Python
    sim_sig = p.executor().process(input_sig_py).data
    np.testing.assert_equal(sim_sig, input_sig_py)

    sim_sig = utils.float_to_fixed_array(sim_sig, 27) << 4
    audio_helpers.write_wav(outfile_py, Fs, sim_sig)

    # Run C
    xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile_c, num_out_channels, n_stages)

    # since gen.sin already generates a quantised input, no need to truncate
    exp_fs, exp_sig = audio_helpers.read_wav(infile)

    # Compare
    out_fs, out_sig_py = audio_helpers.read_wav(outfile_py)
    out_fs, out_sig_c = audio_helpers.read_wav(outfile_c)
    np.testing.assert_equal(out_sig_py, exp_sig[:out_sig_py.shape[0],:]) # Compare Python with input
    np.testing.assert_equal(out_sig_c, exp_sig[:out_sig_c.shape[0],:]) # Compare C with input


INT32_MIN = -(2**31)
INT32_MAX = (-INT32_MIN) - 1
@pytest.mark.parametrize("input, add", [(3, 3),  # input too insignificant, output is shifted
                                        (-3, 0), # negative small numbers truncate in the negative direction
                                        (INT32_MIN, -3),
                                        (INT32_MAX, 3),
                                        (INT32_MAX, -3)])
def test_pipeline_q27(input, add):
    """
    Check that the pipeline operates at q5.27 and outputs saturated Q1.31

    This is done by having a stage which adds a constant to its input. Inputing
    small values will result in the small value being too insgnificant and shifted out.
    and the result is the constant shifted up.

    Check for saturation by adding a constant to large values and checking the output hasn't
    overflowed
    """
    name = f"q27_{input}_{add}.wav"
    infile = "in" + name
    outfile = "out" + name
    n_samps, channels, rate = 1024, 2, 48000

    output = ((input >> HEADROOM_BITS) + add) << HEADROOM_BITS
    output = min(output, INT32_MAX)
    output = max(output, INT32_MIN)

    p, i = Pipeline.begin(channels)
    addn = p.stage(AddN, i, n=add)
    p.set_outputs(addn)

    if input == INT32_MAX:
        app_str = "max"
    elif input == INT32_MIN:
        app_str = "min"
    else:
        app_str = f"{input}"

    app_dir = PKG_DIR / f"test_pipeline_q27_{app_str}"
    os.makedirs(app_dir, exist_ok=True)

    target = "default"
    gen_build(app_dir, p, target)

    sig = np.multiply(np.ones((n_samps, channels), dtype=np.int32), input, dtype=np.int32)
    audio_helpers.write_wav(infile, rate, sig)

    xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, num_out_channels, pipeline_stages=1)

    expected = np.multiply(np.ones((n_samps, channels), dtype=np.int32), output, dtype=np.int32)
    _, out_data = audio_helpers.read_wav(outfile)
    np.testing.assert_equal(expected, out_data)

@pytest.mark.group0
def test_complex_pipeline():
    """
    Generate a multithreaded pipeline and check the output is as expected
    """
    infile = "incomplex.wav"
    outfile = "outcomplex.wav"
    n_samps, channels, rate = 1024, 2, 48000

    p, i = Pipeline.begin(channels)       # ch   0   1
    # this thread has both channels
    a = p.stage(AddN, i, n=1)          #     +1  +1
    a = p.stage(AddN, a, n=1)          #     +2  +2
    a0 = p.stage(AddN, a[:1], n=1)     #     +3  +2
    p.next_thread()
    # this thread has channel 1
    a1 = p.stage(AddN, a[1:], n=1)     #     +3  +3
    a1 = p.stage(AddN, a1, n=1)        #     +3  +4
    a1 = p.stage(AddN, a1, n=1)        #     +3  +5
    p.next_thread()
    # this thread has channel 0
    a0 = p.stage(AddN, a0, n=1)        #     +4  +5
    p.next_thread()
    # this thread has both channels
    a = p.stage(AddN, a0 + a1, n=1)  #     +5  +6

    p.set_outputs(a)
    n_stages = 3  # 2 of the 4 threads are parallel

    app_dir = PKG_DIR / "test_pipeline_complex"
    os.makedirs(app_dir, exist_ok=True)

    target = "default"
    gen_build(app_dir, p, target)

    in_val = 1000
    # expected output is +5 on left, +6 on right, in the Q1.27 format
    expected = (np.array([[5, 6]]*n_samps) + (in_val >> HEADROOM_BITS)) << HEADROOM_BITS
    sig = np.multiply(np.ones((n_samps, channels), dtype=np.int32), in_val, dtype=np.int32)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}/pipeline_test_{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, num_out_channels, n_stages)
    _, out_data = audio_helpers.read_wav(outfile)
    np.testing.assert_equal(expected, out_data)

@pytest.mark.group0
def test_stage_labels():
    """
    Test for the user defined stage labels.
    Compares the autogenerated adsp_instance_id.h file to the stage labels dictionary in the resolved pipeline and
    confirms that they match.
    Also, runs some code on the device to send an internal control command using the define in adsp_instance_id.h
    for the instance ID used in the command and checks if control works as expected.
    """
    p, n_stages = create_pipeline()

    # Autogenerate C code
    with FileLock(build_utils.PIPELINE_BUILD_LOCK):
        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
        target = "default"

        # Check if the adsp_instance_id.h file exists and the labels are present in it
        label_defines_file = BUILD_DIR / "dsp_pipeline" / "adsp_instance_id_auto.h"
        assert label_defines_file.is_file(), f"{label_defines_file} not found"

        resolved_pipe = p.resolve_pipeline()
        labels_py = resolved_pipe["labels"]

        labels_autogen = {}
        with open(label_defines_file, "r") as fp:
            lines = fp.read().splitlines()
            for line in lines:
                m = re.match(r"^\#define\s+(\S+)_stage_index\s+\((\d+)\)", line)
                if m:
                    assert(len(m.groups()) == 2)
                    labels_autogen[m.groups()[0]] = int(m.groups()[1])
        assert(labels_py == labels_autogen), f"stage labels in Python pipeline do not match the stage labels in the autogenerated code"



