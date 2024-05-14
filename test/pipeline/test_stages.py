# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages with 2 inputs and 2 ouputs
"""
import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages.biquad import Biquad
from audio_dsp.stages.cascaded_biquads import CascadedBiquads
from audio_dsp.stages.limiter import LimiterRMS, LimiterPeak, HardLimiterPeak, Clipper
from audio_dsp.stages.noise_gate import NoiseGate
from audio_dsp.stages.noise_suppressor import NoiseSuppressor
from audio_dsp.stages.signal_chain import VolumeControl, FixedGain, Delay
from audio_dsp.stages.compressor import CompressorRMS
from audio_dsp.stages.reverb import Reverb

import audio_dsp.dsp.utils as utils
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np

PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

TEST_FRAME_SIZES=1, #128

fs = 48000
channels = 2  # if this changes need to rewrite test signals

@pytest.fixture(scope="module", params=TEST_FRAME_SIZES)
def frame_size(request):
    return request.param

def generate_ref(sig, ref_module, pipeline_channels, frame_size):
    """
    Process the signal through a python stage of a certain frame size
    """
    sig_flt = np.float64(sig.T) * 2**-31
    signal_frames = utils.frame_signal(sig_flt, frame_size, frame_size)
    out_py = np.zeros((pipeline_channels, sig.shape[0]))

    # run through python bit exact implementation
    for n in range(len(signal_frames)):
        out_py[:, n*frame_size:n*frame_size+frame_size] = ref_module.process_frame_xcore(signal_frames[n])

    # back to int scaling
    out_py_int = out_py * 2**31
    print(frame_size, out_py_int)
    return out_py_int


def do_test(make_p, dut_frame_size):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare

    This test compares the stage running on the device with the simulated
    stage. It runs the simulated stage at all the frame sizes in TEST_FRAME_SIZES
    to ensure that the output of a stage doesn't change when the frame size changes.

    Parameters
    ----------
    make_p: function
        function that takes a frame size and returns a pipeline which
        has that frame size as the input.
    dut_frame_size : int
        The frame size to use for the pipeline that will run on the device.
    """
    dut_p = make_p(dut_frame_size)
    pipeline_channels = len(dut_p.i)
    infile = "instage.wav"
    outfile = "outstage.wav"
    n_samps, rate = 1024, 48000

    generate_dsp_main(dut_p, out_dir = BUILD_DIR / "dsp_pipeline")
    target = "pipeline_test"
    # Build pipeline test executable. This will download xscope_fileio if not present
    build_utils.build(APP_DIR, BUILD_DIR, target)

    sig0 = np.linspace(-2**26, 2**26, n_samps, dtype=np.int32)  << 4 # numbers which should be unmodified through pipeline
                                                                     # data formats


    sig1 = np.linspace(-2**23, 2**23, n_samps, dtype=np.int32)  << 4

    if pipeline_channels == 2:
        sig = np.stack((sig0, sig1), axis=1)
    elif pipeline_channels == 1:
        sig = sig0
        sig = sig.reshape((len(sig), 1))
    else:
        assert False, f"Unsupported number of channels {pipeline_channels}. Test supports 1 or 2 channels"

    audio_helpers.write_wav(infile, rate, sig)

    xe = APP_DIR / f"bin/{target}.xe"
    run_pipeline_xcoreai.run(xe, infile, outfile, pipeline_channels, 1)

    _, out_data = audio_helpers.read_wav(outfile)
    if out_data.ndim == 1:
        out_data = out_data.reshape(len(out_data), 1)

    ref_p = [make_p(s) for s in TEST_FRAME_SIZES]
    out_py_int_all = [generate_ref(sig, p.stages[2].dsp_block, pipeline_channels, fr) for p, fr in zip(ref_p, TEST_FRAME_SIZES)]

    for out_py_int, ref_frame_size in zip(out_py_int_all, TEST_FRAME_SIZES):
        for ch in range(pipeline_channels):
            diff = out_py_int.T[:,ch] - out_data[:, ch]
            print(f"ch {ch}: max diff {max(abs(diff))}")
            sol = (~np.equal(out_py_int.T, out_data)).astype(int)
            indexes = np.flatnonzero(sol)
            print(f"ch {ch}: {len(indexes)} indexes mismatch")
            print(f"ch {ch} mismatching indexes = {indexes}")

        np.testing.assert_equal(out_py_int.T, out_data, err_msg=f"dut frame {dut_frame_size}, ref frame {ref_frame_size}")



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
def _test_biquad(method, args, frame_size):
    """
    Test the biquad stage filters the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            biquad = t.stage(Biquad, p.i)
        p.set_outputs(biquad.o)

        bq_method = getattr(biquad, method)
        if args:
            bq_method(*args)
        else:
            bq_method()
        return p

    do_test(make_p, frame_size)


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
def test_cascaded_biquad(method, args, frame_size):
    """
    Test the biquad stage filters the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            cbiquad = t.stage(CascadedBiquads, p.i)
        p.set_outputs(cbiquad.o)

        cbq_method = getattr(cbiquad, method)
        if args:
            cbq_method(*args)
        else:
            cbq_method()
        return p

    do_test(make_p, frame_size)


def _test_limiter_rms(frame_size):
    """
    Test the limiter stage limits the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            lim = t.stage(LimiterRMS, p.i, label="test_xyz")
        p.set_outputs(lim.o)

        lim.make_limiter_rms(-6, 0.001, 0.1)
        print(f"\n\n\npipeline in test: {p.resolve_pipeline()['configs'][2]}")
        return p

    do_test(make_p, frame_size)


def _test_limiter_peak(frame_size):
    """
    Test the limiter stage limits the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            lim = t.stage(LimiterPeak, p.i)
        p.set_outputs(lim.o)

        lim.make_limiter_peak(-6, 0.001, 0.1)
        return p

    do_test(make_p, frame_size)

def _test_hard_limiter_peak(frame_size):
    """
    Test the limiter stage limits the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            lim = t.stage(HardLimiterPeak, p.i)
        p.set_outputs(lim.o)

        lim.make_hard_limiter_peak(-6, 0.001, 0.1)
        return p

    do_test(make_p, frame_size)

def _test_clipper(frame_size):
    """
    Test the clipper stage clips the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            clip = t.stage(Clipper, p.i)
        p.set_outputs(clip.o)

        clip.make_clipper(-6)
        return p

    do_test(make_p, frame_size)

def _test_compressor(frame_size):
    """
    Test the compressor stage compresses the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            comp = t.stage(CompressorRMS, p.i)
        p.set_outputs(comp.o)

        comp.make_compressor_rms(2, -6, 0.001, 0.1)
        return p

    do_test(make_p, frame_size)

def _test_noise_gate(frame_size):
    """
    Test the noise gate stage gates the noise the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            ng = t.stage(NoiseGate, p.i)
        p.set_outputs(ng.o)

        ng.make_noise_gate(-6, 0.001, 0.1)
        return p

    do_test(make_p, frame_size)

def _test_noise_suppressor(frame_size):
    """
    Test the noise suppressor stage suppress the noise the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            ng = t.stage(NoiseSuppressor, p.i)
        p.set_outputs(ng.o)

        ng.make_noise_suppressor(2, -6, 0.001, 0.1)
        return p

    do_test(make_p, frame_size)

def _test_volume(frame_size):
    """
    Test the volume stage amplifies the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            vol = t.stage(VolumeControl, p.i, gain_dB=-6)
        p.set_outputs(vol.o)
        return p

    do_test(make_p, frame_size)


def _test_fixed_gain(frame_size):
    """
    Test the volume stage amplifies the same in python and C
    """
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            vol = t.stage(FixedGain, p.i)
        p.set_outputs(vol.o)

        vol.set_gain(-6)
        return p

    do_test(make_p, frame_size)

def _test_reverb(frame_size):
    """
    Test Reverb stage
    """
    def make_p(fr):
        reverb_test_channels = 1 # Reverb expects only 1 channel
        p = Pipeline(reverb_test_channels, frame_size=fr)
        with p.add_thread() as t:
            rv = t.stage(Reverb, p.i)
        p.set_outputs(rv.o)
        return p


    do_test(make_p, frame_size)

def _test_delay(frame_size):
    """
    Test Delay stage
    """
    pass
    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        with p.add_thread() as t:
            delay = t.stage(Delay, p.i, max_delay=15, starting_delay=10)
        p.set_outputs(delay.o)
        return p

    do_test(make_p, frame_size)
