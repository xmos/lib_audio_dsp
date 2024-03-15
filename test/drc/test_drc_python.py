# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
import os

import audio_dsp.dsp.drc as drc
import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_gen as gen


def make_noisy_speech():
    hydra_audio_path = os.environ['hydra_audio_PATH']
    filepath = Path(hydra_audio_path, 'acoustic_team_test_audio',
                    'speech', "010_male_female_single-talk_seq.wav")
    sig, fs = sf.read(filepath)
    amp = utils.db2gain(-30)
    noise_sig = gen.pink_noise(fs, len(sig)/fs, amp)

    out_sig = sig + noise_sig
    return out_sig, fs

@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("threshold", [-20, -10, -6, 0])
def test_limiter_peak_attack(fs, at, threshold):
    # Attack time test bads on Figure 2 in Guy McNally's "Dynamic Range Control
    # of Digital Audio Signals"

    # Make a constant signal at 6 dB above the threshold, make 2* length of
    # attack time to keep the test quick
    x = np.ones(int(at*2*fs))
    x[:] = utils.db2gain(threshold + 6)
    t = np.arange(len(x))/fs

    # fixed release time, not that we're going to use it
    lt = drc.limiter_peak(fs, 1, threshold, at, 0.3)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    # do the processing
    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

    # attack time is defined as how long to get within 2 dB of final value,
    # in this case the threshold. Find when we get to this
    thresh_passed = np.argmax(utils.db(env) > threshold)
    sig_3dB = np.argmax(utils.db(y) < threshold + 2)

    measured_at = t[sig_3dB] - t[thresh_passed]
    print("target: %.3f, measured: %.3f" % (at, measured_at))

    # be somewhere vaugely near the spec, attack time definition is variable!
    assert measured_at/at > 0.85
    assert measured_at/at < 1.15


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("rt", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("threshold", [-20, -10, -6, 0])
def test_limiter_peak_release(fs, rt, threshold):
    # Release time test bads on Figure 2 in Guy McNally's "Dynamic Range
    # Control of Digital Audio Signals"

    # Make a step down signal from 6 dB above the threshold to 3 dB below
    # threshold, make 2* length of release time plus 0.5 to keep the test quick
    x = np.ones(int((0.5+rt*2)*fs))
    x[:int(0.5*fs)] = utils.db2gain(threshold + 6)
    x[int(0.5*fs):] = utils.db2gain(threshold - 3)
    t = np.arange(len(x))/fs

    # fixed attack time of 0.01, so should have converged by 0.5s
    lt = drc.limiter_peak(fs, 1, threshold, 0.01, rt)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    # do the processing
    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

    # find when within 3 dB of target value after dropping below the threshold,
    # in this case the original value of 2 dB below the threshold.
    sig_3dB = np.argmax(utils.db(y[int(0.5*fs):]) > threshold - 2 - 3)

    measured_rt = t[sig_3dB]
    print("target: %.3f, measured: %.3f" % (rt, measured_rt))
    print(measured_rt/rt)

    # be somewhere vaugely near the spec
    assert measured_rt/rt > 0.8
    assert measured_rt/rt < 1.2

@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("threshold", [0, -6, -12])
@pytest.mark.parametrize("ratio", (1, 2, 6, np.inf))
@pytest.mark.parametrize("rt", [0.00000001])
@pytest.mark.parametrize("at", [0.00000001])
def test_comp_ratio(fs, at, rt, ratio, threshold):
    # make sure a fast compressor has the same perforance as a limiter
    # over a variety of ratios

    drcut = drc.compressor_rms(fs, 1, ratio, threshold, at, rt)

    signal = gen.log_chirp(fs, (0.1+(rt+at)*2), 1)

    output_xcore = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_int = np.zeros(len(signal))

    # limiter and compressor have 3 outputs
    for n in np.arange(len(signal)):
        output_xcore[n], _, _ = drcut.process_xcore(signal[n])
    drcut.reset_state()
    for n in np.arange(len(signal)):
        output_flt[n], _, _ = drcut.process(signal[n])
    drcut.reset_state()
    for n in np.arange(len(signal)):
        output_int[n], _, _ = drcut.process_int(signal[n])

    # lazy limiter
    ref_signal = np.copy(signal)
    over_thresh = utils.db(ref_signal) > threshold
    ref_signal[over_thresh] *= utils.db2gain((1 - 1/ratio)*(threshold - utils.db(ref_signal[over_thresh])))

    np.testing.assert_allclose(ref_signal, output_flt, atol=3e-16, rtol=0)
    np.testing.assert_allclose(output_flt, output_int, atol=6e-8, rtol=0)
    np.testing.assert_allclose(output_flt, output_xcore, atol=6e-8, rtol=0)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("threshold", [-20, -10, -6, 0])
def comp_vs_limiter(fs, at, threshold):
    # check infinite ratio compressor is a limiter

    # Make a constant signal at 6dB above the threshold, make 2* length of
    # attack time to keep the test quick
    x = np.ones(int(at*2*fs))
    x[:] = utils.db2gain(threshold + 6)
    t = np.arange(len(x))/fs

    rt = 0.3
    comp_type = "rms"
    comp_handle = getattr(drc, "compressor_%s" % comp_type)
    lim_handle = getattr(drc, "limiter_%s" % comp_type)

    comp_thing = comp_handle(fs, 1, threshold, np.inf, at, rt)
    lim_thing = lim_handle(fs, 1, threshold, at, rt)

    y_p = np.zeros_like(x)
    f_p = np.zeros_like(x)
    env_p = np.zeros_like(x)

    # do the processing
    for n in range(len(y_p)):
        y_p[n], f_p[n], env_p[n] = comp_thing.process(x[n])

    y_r = np.zeros_like(x)
    f_r = np.zeros_like(x)
    env_r = np.zeros_like(x) 

    # do the processing
    for n in range(len(y_r)):
        y_r[n], f_r[n], env_r[n] = lim_thing.process(x[n])

    # limiter and infinite ratio compressor should be the same
    np.testing.assert_allclose(utils.db(y_p),
                               utils.db(y_r),
                               atol=0.002)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("threshold", [-20, -10, -6, 0])
def test_peak_vs_rms(fs, at, threshold):
    # check peak and rms converge to same value

    # Make a constant signal at 6dB above the threshold, make 2* length of
    # attack time to keep the test quick
    x = np.ones(int(at*10*fs))
    x[:] = utils.db2gain(threshold + 6)
    t = np.arange(len(x))/fs

    rt = 0.3
    comp_type = "limiter"
    peak_handle = getattr(drc, "%s_peak" % comp_type)
    rms_handle = getattr(drc, "%s_rms" % comp_type)

    peak_thing = peak_handle(fs, 1, threshold, at, rt)
    rms_thing = rms_handle(fs, 1, threshold, at, rt)

    y_p = np.zeros_like(x)
    f_p = np.zeros_like(x)
    env_p = np.zeros_like(x)

    # do the processing
    for n in range(len(y_p)):
        y_p[n], f_p[n], env_p[n] = peak_thing.process(x[n])

    y_r = np.zeros_like(x)
    f_r = np.zeros_like(x)
    env_r = np.zeros_like(x)

    # do the processing
    for n in range(len(y_r)):
        y_r[n], f_r[n], env_r[n] = rms_thing.process(x[n])

    # rms and peak limiter should converge to the same value
    np.testing.assert_allclose(utils.db(y_p[int(fs*at*5):]),
                               utils.db(y_r[int(fs*at*5):]),
                               atol=0.0022)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("at", [0.01])
@pytest.mark.parametrize("threshold", [-10])
def test_sidechain_mono_vs_comp(fs, at, threshold):
    # test a sidechain compressor is the same as a normal compressor
    # when the sidechain signal is the same as the input signal

    ratio = 5

    x = gen.sin(fs, 1, 1, 1)
    t = np.arange(len(x))/fs

    rt = 0.5
    comp_type = "compressor_rms"
    reg_handle = getattr(drc, "%s" % comp_type)
    side_handle = getattr(drc, "%s_sidechain_mono" % comp_type)

    reg_thing = reg_handle(fs, 1, ratio, threshold, at, rt)
    side_thing = side_handle(fs, ratio, threshold, at, rt)

    y_p = np.zeros_like(x)
    f_p = np.zeros_like(x)
    env_p = np.zeros_like(x)

    # do the processing
    for n in range(len(y_p)):
        y_p[n], f_p[n], env_p[n] = reg_thing.process(x[n])

    y_r = np.zeros_like(x)
    f_r = np.zeros_like(x)
    env_r = np.zeros_like(x)

    # do the processing
    for n in range(len(y_r)):
        y_r[n], f_r[n], env_r[n] = side_thing.process(x[n], x[n])

    # rms and peak limiter should converge to the same value
    np.testing.assert_array_equal(y_p, y_r)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("at", [0.01])
@pytest.mark.parametrize("threshold", [-10])
def test_sidechain_stereo(fs, at, threshold):
    # check peak and rms converge to same value

    # Make a constant signal at 6dB above the threshold, make 2* length of
    # attack time to keep the test quick
    x = gen.sin(fs, 1, 1, 1)
    x = np.stack([x, x], axis=0)
    t = np.arange(len(x))/fs

    rt = 0.3
    comp_type = "compressor_rms"
    reg_handle = getattr(drc, "%s_stereo" % comp_type)
    side_handle = getattr(drc, "%s_sidechain_stereo" % comp_type)

    reg_thing = reg_handle(fs, 1, threshold, at, rt)
    side_thing = side_handle(fs, 1, threshold, at, rt)

    y_p = np.zeros_like(x)
    f_p = np.zeros_like(x)
    env_p = np.zeros_like(x)

    # do the processing
    for n in range(len(y_p)):
        y_p[:, n], f_p[:, n], env_p[:, n] = reg_thing.process_channels(x[:, n])

    y_r = np.zeros_like(x)
    f_r = np.zeros_like(x)
    env_r = np.zeros_like(x)

    # do the processing
    for n in range(len(y_r)):
        y_r[:, n], f_r[:, n], env_r[:, n] = side_thing.process_channels(x[:, n], x[:, n])

    # rms and peak limiter should converge to the same value
    np.testing.assert_array_equal(y_p, y_r)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("component_mono, component_stereo, threshold, ratio", [("limiter_peak", "limiter_peak_stereo", -20, None),
                                                                                ("limiter_peak", "limiter_peak_stereo", -6, None),
                                                                                ("compressor_rms", "compressor_rms_stereo", 0, 6),
                                                                                ("compressor_rms", "compressor_rms_stereo", 0, 2),
                                                                                ("compressor_rms_sidechain_mono", "compressor_rms_sidechain_stereo", 0, 6),
                                                                                ("compressor_rms_sidechain_mono", "compressor_rms_sidechain_stereo", 0, 2)])
@pytest.mark.parametrize("rt", [0.2])
@pytest.mark.parametrize("at", [0.001])
def test_mono_vs_stereo(fs, component_mono, component_stereo, at, rt, threshold, ratio):
    # test the mono and stereo components have the same perforamnce when
    # fed a dual mono signal

    signal = []
    lenght = 0.1 + (rt + at) * 2
    f = 997
    signal.append(gen.sin(fs, lenght, f, 1))
    signal.append(gen.sin(fs, lenght, f, 1))
    signal = np.stack(signal, axis=0)

    if "sidechain" in component_stereo:
        sidechain_signal = np.zeros_like(signal)
        sidechain_signal[:, len(signal)//2:] = 1

    stereo_component_handle = getattr(drc, component_stereo)
    mono_component_handle = getattr(drc, component_mono)
    if ratio is not None:
        drc_s = stereo_component_handle(fs, ratio, threshold, at, rt)
        drc_m = mono_component_handle(fs, 1, ratio, threshold, at, rt)

    else:
        drc_s = stereo_component_handle(fs, threshold, at, rt)
        drc_m = mono_component_handle(fs, 1, threshold, at, rt)


    output_xcore_s = np.zeros(signal.shape)
    output_flt_s = np.zeros(signal.shape)
    output_int_s = np.zeros(signal.shape)

    if "sidechain" in component_stereo:
        for n in np.arange(signal.shape[1]):
            output_xcore_s[:, n], _, _ = drc_s.process_channels_xcore(signal[:, n], sidechain_signal[:, n])
        drc_s.reset_state()
        for n in np.arange(signal.shape[1]):
            output_flt_s[:, n], _, _ = drc_s.process_channels(signal[:, n], sidechain_signal[:, n])
        drc_s.reset_state()
        for n in np.arange(signal.shape[1]):
            output_int_s[:, n], _, _ = drc_s.process_channels_int(signal[:, n], sidechain_signal[:, n])
    else:
        for n in np.arange(signal.shape[1]):
            output_xcore_s[:, n], _, _ = drc_s.process_channels_xcore(signal[:, n])
        drc_s.reset_state()
        for n in np.arange(signal.shape[1]):
            output_flt_s[:, n], _, _ = drc_s.process_channels(signal[:, n])
        drc_s.reset_state()
        for n in np.arange(signal.shape[1]):
            output_int_s[:, n], _, _ = drc_s.process_channels_int(signal[:, n])
    


    output_xcore_m = np.zeros(signal.shape)
    output_flt_m = np.zeros(signal.shape)
    output_int_m = np.zeros(signal.shape)

    # write mono signal to both output channels, makes comparison to stereo easier
    if "sidechain" in component_mono:
        for n in np.arange(signal.shape[1]):
            output_xcore_m[:, n], _, _  = drc_m.process_xcore(signal[0, n], sidechain_signal[0, n])
        drc_m.reset_state()
        for n in np.arange(signal.shape[1]):
            output_flt_m[:, n], _, _ = drc_m.process(signal[0, n], sidechain_signal[0, n])
        drc_m.reset_state()
        for n in np.arange(signal.shape[1]):
            output_int_m[:, n], _, _ = drc_m.process_int(signal[0, n], sidechain_signal[0, n])
    else:
        for n in np.arange(signal.shape[1]):
            output_xcore_m[:, n], _, _  = drc_m.process_xcore(signal[0, n])
        drc_m.reset_state()
        for n in np.arange(signal.shape[1]):
            output_flt_m[:, n], _, _ = drc_m.process(signal[0, n])
        drc_m.reset_state()
        for n in np.arange(signal.shape[1]):
            output_int_m[:, n], _, _ = drc_m.process_int(signal[0, n])


    # check stereo channels are the same
    np.testing.assert_array_equal(output_flt_s[0], output_flt_s[1])
    np.testing.assert_array_equal(output_int_s[0], output_int_s[1])
    np.testing.assert_array_equal(output_xcore_s[0], output_xcore_s[1])

    # check stereo equals mono
    np.testing.assert_array_equal(output_flt_s, output_flt_m)
    np.testing.assert_array_equal(output_int_s, output_int_m)
    np.testing.assert_array_equal(output_xcore_s, output_xcore_m)


def test_noise_gate():
    # test the noise gate performance on noisy speech
    signal, fs = make_noisy_speech()

    test_len = int(6*fs)
    signal = signal[:test_len]

    drcut = drc.noise_gate(fs, 1, -30, 0, 0.2)

    output_xcore = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_int = np.zeros(len(signal))

    # noise gate has 3 outputs
    for n in np.arange(len(signal)):
        output_xcore[n], _, _ = drcut.process_xcore(signal[n])
    drcut.reset_state()
    for n in np.arange(len(signal)):
        output_flt[n], _, _ = drcut.process(signal[n])
    drcut.reset_state()
    for n in np.arange(len(signal)):
        output_int[n], _, _ = drcut.process_int(signal[n])
    sf.write("noise_gate_test_in.wav", signal, fs)
    sf.write("noise_gate_test_out.wav", output_flt, fs)

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055

        error_int = np.abs(utils.db(output_int[top_half])-utils.db(output_flt[top_half]))
        mean_error_int = utils.db(np.nanmean(utils.db2gain(error_int)))
        assert mean_error_int < 0.055


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("component, threshold, ratio", [("limiter_peak", 0, None),
                                                         ("limiter_rms", 0, None),
                                                         ("compressor_rms", 0, 6),
                                                         ("compressor_rms", 0, 2),
                                                         ("noise_gate", -1000, None)])
@pytest.mark.parametrize("rt", [0.2, 0.3, 0.5])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.1])
def test_drc_component_bypass(fs, component, at, rt, threshold, ratio):
    # test that a drc component is bit exact when the signal is below
    # the threshold (or above in the case of a noise gate).
    if component == "noise_gate":
        #TODO fixme
        pytest.xfail("suspected float32 issue for noise gate")
    # check that a 24b quantized chirp is bit exact if it's below the threshold
    component_handle = getattr(drc, component)

    if threshold is not None:
        if ratio is not None:
            drcut = component_handle(fs, 1, ratio, threshold, at, rt)
        else:
            drcut = component_handle(fs, 1, threshold, at, rt)
    else:
        drcut = component_handle(fs, 1, at, rt)

    signal = gen.log_chirp(fs, (0.1+(rt+at)*2), 1)

    output_xcore = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_int = np.zeros(len(signal))

    if "envelope" in component:
        # envelope detector has 1 output
        for n in np.arange(len(signal)):
            output_xcore[n] = drcut.process_xcore(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_flt[n] = drcut.process(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_int[n] = drcut.process_int(signal[n])
    else:
        # limiter and compressor have 3 outputs
        for n in np.arange(len(signal)):
            output_xcore[n], _, _ = drcut.process_xcore(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_flt[n], _, _ = drcut.process(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_int[n], _, _ = drcut.process_int(signal[n])

    np.testing.assert_array_equal(signal, output_flt)
    np.testing.assert_array_equal(signal, output_int)
    np.testing.assert_array_equal(signal, output_xcore)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("component, threshold, ratio", [("limiter_peak", -20, None),
                                                         ("limiter_peak", 6, None),
                                                         ("limiter_rms", -20, None),
                                                         ("limiter_rms", 6, None),
                                                         ("envelope_detector_peak", None, None),
                                                         ("envelope_detector_rms", None, None),
                                                         ("compressor_rms", -20, 6),
                                                         ("compressor_rms", -20, 2),
                                                         ("compressor_rms", 6, 6),
                                                         ("compressor_rms", 6, 2),
                                                         ("noise_gate", -20, None)])
@pytest.mark.parametrize("rt", [0.05, 0.1, 0.2, 0.5, 3.0])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
def test_drc_component(fs, component, at, rt, threshold, ratio):
    # test the process_ functions of the drc components
    component_handle = getattr(drc, component)

    if threshold is not None:
        if ratio is not None:
            drcut = component_handle(fs, 1, ratio, threshold, at, rt)
        else:
            drcut = component_handle(fs, 1, threshold, at, rt)
    else:
        drcut = component_handle(fs, 1, at, rt)

    signal = gen.log_chirp(fs, (0.1+(rt+at)*2), 1)
    len_sig = len(signal)

    if threshold is not None:
        # If we are a limiter or compressor, have first half of signal above
        # the threshold and second half below
        signal[:len_sig//2] *= utils.db2gain(threshold + 6)
        signal[len_sig//2:] *= utils.db2gain(threshold - 3)
    else:
        # if we are an envelope detector, amplitude modulate with a sin to give
        # something to follow
        t = np.arange(len(signal))/fs
        signal *= np.sin(t*2*np.pi*0.5)

    output_xcore = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_int = np.zeros(len(signal))

    if "envelope" in component:
        # envelope detector has 1 output
        for n in np.arange(len(signal)):
            output_xcore[n] = drcut.process_xcore(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_flt[n] = drcut.process(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_int[n] = drcut.process_int(signal[n])
    else:
        # limiter and compressor have 3 outputs
        for n in np.arange(len(signal)):
            output_xcore[n], _, _ = drcut.process_xcore(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_flt[n], _, _ = drcut.process(signal[n])
        drcut.reset_state()
        for n in np.arange(len(signal)):
            output_int[n], _, _ = drcut.process_int(signal[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055

        error_int = np.abs(utils.db(output_int[top_half])-utils.db(output_flt[top_half]))
        mean_error_int = utils.db(np.nanmean(utils.db2gain(error_int)))
        assert mean_error_int < 0.055


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("component, threshold, ratio", [("limiter_peak", -20, None),
                                                         ("limiter_peak", 6, None),
                                                         ("limiter_rms", -20, None),
                                                         ("limiter_rms", 6, None),
                                                         ("envelope_detector_peak", None, None),
                                                         ("envelope_detector_rms", None, None),
                                                         ("compressor_rms", -20, 6),
                                                         ("compressor_rms", -20, 2),
                                                         ("compressor_rms", 6, 6),
                                                         ("compressor_rms", 6, 2),
                                                         ("noise_gate", -20, None)])
@pytest.mark.parametrize("rt", [0.2, 0.3, 0.5])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
def test_drc_component_frames(fs, component, at, rt, threshold, ratio, n_chans):
    # test the process_frame functions of the drc components

    component_handle = getattr(drc, component)

    if threshold is not None:
        if ratio is not None:
            drcut = component_handle(fs, n_chans, ratio, threshold, at, rt)
        else:
            drcut = component_handle(fs, n_chans, threshold, at, rt)
    else:
        drcut = component_handle(fs, n_chans, at, rt)

    signal = gen.log_chirp(fs, (0.1+(rt+at)*2), 1)
    len_sig = len(signal)
    if threshold is not None:
        signal[:len_sig//2] *= utils.db2gain(threshold + 6)
        signal[len_sig//2:] *= utils.db2gain(threshold - 3)
    else:
        t = np.arange(len(signal))/fs
        signal *= np.sin(t*2*np.pi*0.5)

    signal = np.tile(signal, [n_chans, 1])
    frame_size = 1
    signal_frames = utils.frame_signal(signal, frame_size, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)

    for n in range(len(signal_frames)):
        output_int[:, n:n+frame_size] = drcut.process_frame_xcore(signal_frames[n])
    drcut.reset_state()
    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = drcut.process_frame(signal_frames[n])

    assert np.all(output_int[0, :] == output_int)
    assert np.all(output_flt[0, :] == output_flt)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("component, threshold, ratio", [("limiter_peak_stereo", -20, None),
                                                         ("limiter_peak_stereo", -6, None),
                                                         ("compressor_rms_stereo", 0, 6),
                                                         ("compressor_rms_stereo", 0, 2)])
@pytest.mark.parametrize("rt", [0.2, 0.3, 0.5])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.1])
def test_stereo_components(fs, component, at, rt, threshold, ratio):
    # test the process_channels functions of the stereo drc components

    component_handle = getattr(drc, component)
    if ratio is not None:
        drcut = component_handle(fs, ratio, threshold, at, rt)
    else:
        drcut = component_handle(fs, threshold, at, rt)

    signal = []
    lenght = 0.1 + (rt + at) * 2
    f = 997
    signal.append(gen.sin(fs, lenght, f, 1))
    signal.append(gen.sin(fs, lenght, f, 0.5))
    signal = np.stack(signal, axis=0).astype(np.float32)

    output_xcore = np.zeros(signal.shape, dtype=np.float32)
    output_flt = np.zeros(signal.shape, dtype=np.float32)
    output_int = np.zeros(signal.shape, dtype=np.float32)

    for n in np.arange(signal.shape[1]):
        output_xcore[:, n], _, _ = drcut.process_channels_xcore(signal[:, n])
    drcut.reset_state()
    for n in np.arange(signal.shape[1]):
        output_flt[:, n], _, _ = drcut.process_channels(signal[:, n])
    drcut.reset_state()
    for n in np.arange(signal.shape[1]):
        output_int[:, n], _, _ = drcut.process_channels_int(signal[:, n])
    
    error_flt = np.abs(utils.db(output_xcore)-utils.db(output_flt))
    mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
    assert mean_error_flt < 0.055

    error_int = np.abs(utils.db(output_int)-utils.db(output_flt))
    mean_error_int = utils.db(np.nanmean(utils.db2gain(error_int)))
    assert mean_error_int < 0.055


# TODO more RMS limiter tests
# TODO hard limiter test
# TODO envelope detector tests
# TODO compressor tests

if __name__ == "__main__":
    # test_drc_component(48000, "limiter_peak", 1, 1, 1)
    # test_limiter_peak_attack(48000, 0.1, -10)
    # comp_vs_limiter(48000, 0.001, 0)
    # test_comp_ratio(48000, 0.00000001, 0.00000001, 2, -10)
    test_mono_vs_stereo(48000, "limiter_peak", "limiter_peak_stereo", 0.001, 0.01, -6)
