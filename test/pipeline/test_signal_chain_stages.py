# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages.signal_chain stages that have a different 
number of inputs and outputs
"""
import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages.signal_chain import Adder, Subtractor, Mixer, Switch
from audio_dsp.stages.compressor_sidechain import CompressorSidechain

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_chain as sc
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np


PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

fs = 48000

def do_test(p):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare
    """
    infile = "inadder.wav"
    outfile = "outadder.wav"
    n_samps, rate = 1024, 48000

    # use the python dsp_block as a reference implementation
    ref_module = p.stages[2].dsp_block

    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    sig0 = np.linspace(-2**26, 2**26, n_samps, dtype=np.int32)  << 4 # numbers which should be unmodified through pipeline
                                                                     # data formats

    if type(ref_module) == sc.subtractor:
        # don't overflow output by doing -1 - 0.125
        sig1 = np.linspace(-2**23, 2**23, n_samps, dtype=np.int32)  << 4
    else:
        sig1 = np.linspace(2**23, -2**23, n_samps, dtype=np.int32)  << 4
    sig = np.stack((sig0, sig1), axis=1)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, 1, 1)

    _, out_data = audio_helpers.read_wav(outfile)

    # convert to float scaling and make frames
    frame_size = 1
    sig_flt = np.float64(sig.T) * 2**-31
    signal_frames = utils.frame_signal(sig_flt, frame_size, frame_size)
    out_py = np.zeros((1, sig.shape[0]))
    
    # run through python bit exact implementation
    for n in range(len(signal_frames)):
        out_py[:, n:n+frame_size] = ref_module.process_frame_xcore(signal_frames[n])

    # back to int scaling
    out_py_int = out_py * 2**31

    np.testing.assert_equal(out_py_int[0], out_data)


def test_adder():
    """
    Test the adder stage adds the same in python and C
    """
    channels = 2
    p = Pipeline(channels)
    with p.add_thread() as t:
        adder = t.stage(Adder, p.i)
    p.set_outputs(adder.o)

    do_test(p)



def test_subtractor():
    """
    Test the subtractor stage adds the same in python and C
    """
    channels = 2
    p = Pipeline(channels)
    with p.add_thread() as t:
        adder = t.stage(Subtractor, p.i)
    p.set_outputs(adder.o)

    do_test(p)


@pytest.mark.parametrize("gain", ([-6, 0]))
def test_mixer(gain):
    """
    Test the mixer stage adds the same in python and C
    """
    channels = 2
    p = Pipeline(channels)
    with p.add_thread() as t:
        adder = t.stage(Mixer, p.i).set_gain(gain)
    p.set_outputs(adder.o)

    do_test(p)


def test_compressor_sidechain():
    """
    Test the compressor stage compresses the same in python and C
    """
    channels = 2
    p = Pipeline(channels)
    with p.add_thread() as t:
        comp = t.stage(CompressorSidechain, p.i)
    p.set_outputs(comp.o)

    comp.make_compressor_sidechain(2, -6, 0.001, 0.1)

    do_test(p)

@pytest.mark.parametrize("position", ([0, 1]))
def test_switch(position):
    """
    Test the mixer stage adds the same in python and C
    """
    channels = 2
    p = Pipeline(channels)
    with p.add_thread() as t:
        switch_dsp = t.stage(Switch, p.i).move_switch(position)
    p.set_outputs(switch_dsp.o)

    do_test(p)


if __name__ == "__main__":
    test_subtractor(0)
