# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages.signal_chain stages that have a different
number of inputs and outputs
"""
import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages.signal_chain import Adder, Subtractor, Mixer, Switch, SwitchStereo
from audio_dsp.stages.compressor_sidechain import CompressorSidechain

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_chain as sc
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np
import soundfile as sf
import os
import shutil
from filelock import FileLock

PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

def do_test(p, folder_name, n_outs=1):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare
    """

    app_dir = PKG_DIR / folder_name
    os.makedirs(app_dir, exist_ok=True)

    infile = app_dir / "inadder.wav"
    outfile = app_dir / "outadder.wav"
    n_samps, rate = 1024, 48000

    # use the Python dsp_block as a reference implementation
    ref_module = p.stages[2].dsp_block
    with FileLock(build_utils.PIPELINE_BUILD_LOCK):

        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline_initialized")
        target = "default"

        # Build pipeline test executable. This will download xscope_fileio if not present
        build_utils.build(APP_DIR, BUILD_DIR, target)
        os.makedirs(app_dir / "bin", exist_ok=True)
        shutil.copytree(APP_DIR / "bin", app_dir / "bin", dirs_exist_ok=True)

    sig0 = np.linspace(-2**26, 2**26, n_samps, dtype=np.int32)  << 4 # numbers which should be unmodified through pipeline
                                                                     # data formats

    if type(ref_module) == sc.subtractor:
        # don't overflow output by doing -1 - 0.125
        sig1 = np.linspace(-2**23, 2**23, n_samps, dtype=np.int32)  << 4
    else:
        sig1 = np.linspace(2**23, -2**23, n_samps, dtype=np.int32)  << 4

    if type(ref_module) == sc.switch_stereo:
                sig = np.stack((sig0, sig0, sig1, sig1), axis=1)
    else:
        sig = np.stack((sig0, sig1), axis=1)
        

    audio_helpers.write_wav(infile, rate, sig)

    xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, n_outs, 1)

    out_data, _ = sf.read(outfile, always_2d=True)

    # convert to float scaling and make frames
    frame_size = 1
    sig_flt = np.float64(sig.T) * 2**-31
    signal_frames = utils.frame_signal(sig_flt, frame_size, frame_size)
    out_py = np.zeros((n_outs, sig.shape[0]))

    # run through Python bit exact implementation
    for n in range(len(signal_frames)):
        out_py[:, n*frame_size:(n+1)*frame_size] = ref_module.process_frame_xcore(signal_frames[n])

    # back to int scaling
    out_py_int = out_py * 2**31

    np.testing.assert_equal(out_py, out_data.T)

@pytest.mark.group0
def test_adder():
    """
    Test the adder stage adds the same in Python and C
    """
    channels = 2
    p = Pipeline(channels)
    adder = p.stage(Adder, p.i)
    p.set_outputs(adder)

    do_test(p, "adder")



def test_subtractor():
    """
    Test the subtractor stage adds the same in Python and C
    """
    channels = 2
    p = Pipeline(channels)
    adder = p.stage(Subtractor, p.i)
    p.set_outputs(adder)

    do_test(p, "subtractor")

@pytest.mark.group0
@pytest.mark.parametrize("gain", ([-6, 0]))
def test_mixer(gain):
    """
    Test the mixer stage adds the same in Python and C
    """
    channels = 2
    p = Pipeline(channels)
    adder = p.stage(Mixer, p.i, "a")
    p["a"].set_gain(gain)
    p.set_outputs(adder)

    do_test(p, f"mixer_{gain}")

@pytest.mark.group0
def test_compressor_sidechain():
    """
    Test the compressor stage compresses the same in Python and C
    """
    channels = 2
    p = Pipeline(channels)
    comp = p.stage(CompressorSidechain, p.i, "c")
    p.set_outputs(comp)

    p["c"].make_compressor_sidechain(2, -6, 0.001, 0.1)

    do_test(p, "comp_side")

@pytest.mark.parametrize("position", ([0, 1]))
def test_switch(position):
    """
    Test the mixer stage adds the same in Python and C
    """
    channels = 2
    p = Pipeline(channels)
    switch_dsp = p.stage(Switch, p.i, "s")
    p["s"].move_switch(position)
    p.set_outputs(switch_dsp)

    do_test(p, f"switch_{position}")


@pytest.mark.parametrize("position", ([0, 1]))
def test_switch_stereo(position):
    """
    Test the mixer stage adds the same in Python and C
    """
    channels = 4
    p = Pipeline(channels)
    switch_dsp = p.stage(SwitchStereo, p.i, "s")
    p["s"].move_switch(position)
    p.set_outputs(switch_dsp)

    do_test(p, f"switchstereo_{position}", n_outs=2)

if __name__ == "__main__":
    test_switch_stereo(0)