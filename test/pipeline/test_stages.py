# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages with 2 inputs and 2 ouputs
"""
import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages.biquad import Biquad
from audio_dsp.stages.cascaded_biquads import CascadedBiquads
from audio_dsp.stages.limiter import LimiterRMS, LimiterPeak
from audio_dsp.stages.noise_gate import NoiseGate
from audio_dsp.stages.noise_suppressor import NoiseSuppressor
from audio_dsp.stages.signal_chain import VolumeControl, FixedGain
from audio_dsp.stages.compressor import CompressorRMS

import audio_dsp.dsp.utils as utils
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np

PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

fs = 48000
channels = 2  # if this changes need to rewrite test signals


def do_test(p):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare
    """
    infile = "instage.wav"
    outfile = "outstage.wav"
    n_samps, rate = 1024, 48000
    # use the python dsp_block as a reference implementation
    ref_module = p.stages[2].dsp_block

    generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    sig0 = np.linspace(-2**26, 2**26, n_samps, dtype=np.int32)  << 4 # numbers which should be unmodified through pipeline
                                                                     # data formats


    sig1 = np.linspace(-2**23, 2**23, n_samps, dtype=np.int32)  << 4

    sig = np.stack((sig0, sig1), axis=1)
    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, channels, 1)

    _, out_data = audio_helpers.read_wav(outfile)

    # convert to float scaling and make frames
    frame_size = 1
    sig_flt = np.float64(sig.T) * 2**-31
    signal_frames = utils.frame_signal(sig_flt, frame_size, frame_size)
    out_py = np.zeros((channels, sig.shape[0]))
    
    # run through python bit exact implementation
    for n in range(len(signal_frames)):
        out_py[:, n:n+frame_size] = ref_module.process_frame_xcore(signal_frames[n])

    # back to int scaling
    out_py_int = out_py * 2**31

    np.testing.assert_equal(out_py_int.T, out_data)


@pytest.mark.parametrize("method, args", [("make_bypass", None),
                                          ("make_lowpass", [1000, 0.707]),
                                          ("make_highpass", [1000, 0.707]),
                                          ("make_bandpass", [1000, 0.707]),
                                          ("make_bandstop", [1000, 0.707]),
                                          ("make_notch", [1000, 0.707]),
                                          ("make_allpass", [1000, 0.707]),
                                          ("make_peaking", [1000, 0.707, -6]),
                                          ("make_constant_q", [1000, 0.707, -6]),
                                          ("make_lowshelf", [1000, 0.707, -6]),
                                          ("make_highshelf", [1000, 0.707, -6]),
                                          ("make_linkwitz", [200, 0.707, 180, 0.707])])
def test_biquad(method, args):
    """
    Test the biquad stage filters the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        biquad = t.stage(Biquad, p.i)
    p.set_outputs(biquad.o)

    bq_method = getattr(biquad, method)
    if args:
        bq_method(*args)
    else:
        bq_method()

    do_test(p)


filter_spec = [['lowpass', fs*0.4, 0.707],
                ['highpass', fs*0.001, 1],
                ['peaking', fs*1000/48000, 5, 10],
                ['constant_q', fs*500/48000, 1, -10],
                ['notch', fs*2000/48000, 1],
                ['lowshelf', fs*200/48000, 1, 3],
                ['highshelf', fs*5000/48000, 1, -2],
                ['gain', -2]]
@pytest.mark.parametrize("method, args", [("make_butterworth_highpass", [8, 1000]),
                                          ("make_butterworth_lowpass", [8, 1000]),
                                          ("make_parametric_eq", [filter_spec]),])
def test_cascaded_biquad(method, args):
    """
    Test the biquad stage filters the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        cbiquad = t.stage(CascadedBiquads, p.i)
    p.set_outputs(cbiquad.o)

    cbq_method = getattr(cbiquad, method)
    if args:
        cbq_method(*args)
    else:
        cbq_method()

    do_test(p)


def test_limiter_rms():
    """
    Test the limiter stage limits the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        lim = t.stage(LimiterRMS, p.i)
    p.set_outputs(lim.o)

    lim.make_limiter_rms(-6, 0.001, 0.1)

    do_test(p)


def test_limiter_peak():
    """
    Test the limiter stage limits the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        lim = t.stage(LimiterPeak, p.i)
    p.set_outputs(lim.o)

    lim.make_limiter_peak(-6, 0.001, 0.1)

    do_test(p)

def test_compressor():
    """
    Test the compressor stage compresses the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        comp = t.stage(CompressorRMS, p.i)
    p.set_outputs(comp.o)

    comp.make_compressor_rms(2, -6, 0.001, 0.1)

    do_test(p)

def test_noise_gate():
    """
    Test the noise gate stage gates the noise the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        ng = t.stage(NoiseGate, p.i)
    p.set_outputs(ng.o)

    ng.make_noise_gate(-6, 0.001, 0.1)

    do_test(p)

def test_noise_suppressor():
    """
    Test the noise suppressor stage suppress the noise the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        ng = t.stage(NoiseSuppressor, p.i)
    p.set_outputs(ng.o)

    ng.make_noise_suppressor(-6, 0.001, 0.1, 1)

    do_test(p)

def test_volume():
    """
    Test the volume stage amplifies the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        vol = t.stage(VolumeControl, p.i, gain_dB=-6)
    p.set_outputs(vol.o)

    do_test(p)


def test_fixed_gain():
    """
    Test the volume stage amplifies the same in python and C
    """
    p = Pipeline(channels)
    with p.add_thread() as t:
        vol = t.stage(FixedGain, p.i)
    p.set_outputs(vol.o)

    vol.set_gain(-6)

    do_test(p)


if __name__ == "__main__":
    test_noise_suppressor()
    test_biquad("make_lowpass", [1000, 0.707])
    test_biquad("make_lowpass", [1000, 0.707])
