# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest
import numpy as np
import random
import scipy.signal as spsig

import audio_dsp.dsp.cascaded_biquads as cbq
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils


def saturation_test(filter: cbq.cascaded_biquads_8, fs):

    signal = 2**(np.arange(0, 31.5, 0.5)) - 1
    signal = np.repeat(signal, 2)
    signal[::2] *= -1

    # # used for lib_xcore_math biquad test
    # sigint = (np.round(signal).astype(np.int32))
    # np.savetxt("sig.csv", sigint, fmt="%i", delimiter=",")

    signal *= (2**-31)



    output_int = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_vpu = np.zeros(len(signal))

    # for n in np.arange(len(signal)):
    #     output_int[n] = filter.process_int(signal[n])
    # filter.reset_state()
    # for n in np.arange(len(signal)):
    #     output_flt[n] = filter.process(signal[n])
    filter.reset_state()
    for n in np.arange(len(signal)):
        output_vpu[n] = filter.process_xcore(signal[n])

    # # reference result for lib_xcore_math test
    # vpu_int = (np.round(output_vpu * 2**31).astype(np.int32))
    # np.savetxt("out.csv", vpu_int, fmt="%i", delimiter=",")

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_int[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055
        error_vpu = np.abs(utils.db(output_int[top_half])-utils.db(output_vpu[top_half]))
        mean_error_vpu = utils.db(np.nanmean(utils.db2gain(error_vpu)))
        assert mean_error_vpu < 0.05


def chirp_filter_test(filter: cbq.cascaded_biquads_8, fs):
    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)

    output_int = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_vpu = np.zeros(len(signal))

    for n in np.arange(len(signal)):
        output_int[n] = filter.process_int(signal[n])
    filter.reset_state()
    for n in np.arange(len(signal)):
        output_flt[n] = filter.process(signal[n])
    filter.reset_state()
    for n in np.arange(len(signal)):
        output_vpu[n] = filter.process_xcore(signal[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_int[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055
        error_vpu = np.abs(utils.db(output_int[top_half])-utils.db(output_vpu[top_half]))
        mean_error_vpu = utils.db(np.nanmean(utils.db2gain(error_vpu)))
        assert mean_error_vpu < 0.055


@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
@pytest.mark.parametrize("n_filters", [1, 3, 5, 8])
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11])
def test_peq(fs, n_filters, seed):
    # a list of some sensible filters, use them in  random order
    filter_spec = [['lowpass', fs*0.4, 0.707],
                   ['highpass', fs*0.001, 1],
                   ['peaking', fs*1000/48000, 5, 10],
                   ['constant_q', fs*500/48000, 1, -10],
                   ['notch', fs*2000/48000, 1],
                   ['lowshelf', fs*200/48000, 1, 3],
                   ['highshelf', fs*5000/48000, 1, -2],
                   ['bypass'],
                   ['gain', -2]]
    random.Random(seed**n_filters*int(fs/1000)).shuffle(filter_spec)
    filter_spec = filter_spec[:n_filters]
    peq = cbq.parametric_eq_8band(fs, 1, filter_spec)
    chirp_filter_test(peq, fs)



@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_peq_saturation(fs):
    # a list of some sensible filters, use them in  random order
    filter_spec = [['notch', fs*0.05, 1],
                   ['notch', fs*0.10, 1],
                   ['notch', fs*0.15, 1],
                   ['notch', fs*0.20, 1],
                   ['notch', fs*0.25, 1],
                   ['notch', fs*0.30, 1],
                   ['notch', fs*0.35, 1],
                   ['lowshelf', fs*1000/48000, 1, 3]]

    peq = cbq.parametric_eq_8band(fs, 1, filter_spec, Q_sig=31)
    saturation_test(peq, fs)


@pytest.mark.parametrize("filter_type", ["lowpass",
                                         "highpass"])
@pytest.mark.parametrize("f", [20, 100, 1000, 10000, 20000])
@pytest.mark.parametrize("order", [4, 8, 16])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_nth_butterworth(filter_type, f, order, fs):
    f = np.min([f, fs/2*0.95])
    if filter_type == "lowpass":
        if f < 50:
            return
        if f <= 100 and fs > 100000 and order >= 8:
            return
        filter = cbq.butterworth_lowpass(fs, 1, order, f)

    elif filter_type == "highpass":
        if f > 10000:
            return
        if f < 50 and fs > 100000:
            f = 30
        filter = cbq.butterworth_highpass(fs, 1, order, f)

    chirp_filter_test(filter, fs)

    # compare against scipy butterworth response
    w, h = filter.freq_response(1024*32)
    sos = spsig.butter(order, f, fs=fs, btype=filter_type[:-4], output='sos')

    w_ref, h_ref = spsig.sosfreqz(sos, worN=1024*32)

    top_half = utils.db(h) > -50
    err = np.abs(utils.db(h[top_half]) - utils.db(h_ref[top_half]))
    mean_error = utils.db(np.nanmean(utils.db2gain(err)))

    assert mean_error < 0.1

    print(mean_error)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("n_filters", [1, 3, 5, 8])
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
@pytest.mark.parametrize("q_format", [27, 31])
def test_peq_frame(fs, n_filters, seed, n_chans, q_format):
    # a list of some sensible filters, use them in  random order
    filter_spec = [['lowpass', fs*0.4, 0.707],
                   ['highpass', fs*0.001, 1],
                   ['peaking', fs*1000/48000, 5, 10],
                   ['constant_q', fs*500/48000, 1, -10],
                   ['notch', fs*2000/48000, 1],
                   ['lowshelf', fs*200/48000, 1, 3],
                   ['highshelf', fs*5000/48000, 1, -2],
                   ['bypass'],
                   ['gain', -2]]
    random.Random(seed**n_filters*int(fs/1000)).shuffle(filter_spec)
    filter_spec = filter_spec[:n_filters]
    peq = cbq.parametric_eq_8band(fs, n_chans, filter_spec, Q_sig=q_format)

    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)
    output_vpu = np.zeros_like(signal)
    frame_size = 1
    for n in range(len(signal_frames)):
        output_int[:, n*frame_size:(n+1)*frame_size] = peq.process_frame_int(signal_frames[n])
    assert np.all(output_int[0, :] == output_int)
    peq.reset_state()

    for n in range(len(signal_frames)):
        output_flt[:, n*frame_size:(n+1)*frame_size] = peq.process_frame(signal_frames[n])
    assert np.all(output_flt[0, :] == output_flt)
    peq.reset_state()

    for n in range(len(signal_frames)):
        output_vpu[:, n*frame_size:(n+1)*frame_size] = peq.process_frame_xcore(signal_frames[n])
    assert np.all(output_vpu[0, :] == output_vpu)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("n_filters", [10, 14, 16])
@pytest.mark.parametrize("n_chans, seed", [[1, 1],
                                          [2, 2],
                                          [4, 3],])
@pytest.mark.parametrize("q_format", [27, 31])
def test_peq16_frame(fs, n_filters, seed, n_chans, q_format):
    # a list of some sensible filters, use them in  random order
    filter_spec = [['lowpass', fs*0.4, 0.707],
                   ['highpass', fs*0.001, 1],
                   ['peaking', fs*1000/48000, 5, 10],
                   ['constant_q', fs*500/48000, 1, -10],
                   ['notch', fs*2000/48000, 1],
                   ['lowshelf', fs*200/48000, 1, 3],
                   ['highshelf', fs*5000/48000, 1, -2],
                   ['bypass'],
                   ['gain', -2]]
    # random.Random(seed**n_filters*int(fs/1000)).shuffle(filter_spec)
    filter_spec_new = random.Random(seed**n_filters*int(fs/1000)).choices(filter_spec, k=n_filters)
    peq = cbq.parametric_eq_16band(fs, n_chans, filter_spec_new, Q_sig=q_format)

    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)
    output_vpu = np.zeros_like(signal)
    frame_size = 1
    for n in range(len(signal_frames)):
        output_int[:, n*frame_size:(n+1)*frame_size] = peq.process_frame_int(signal_frames[n])
    assert np.all(output_int[0, :] == output_int)
    peq.reset_state()

    for n in range(len(signal_frames)):
        output_flt[:, n*frame_size:(n+1)*frame_size] = peq.process_frame(signal_frames[n])
    assert np.all(output_flt[0, :] == output_flt)
    peq.reset_state()

    for n in range(len(signal_frames)):
        output_vpu[:, n*frame_size:(n+1)*frame_size] = peq.process_frame_xcore(signal_frames[n])
    assert np.all(output_vpu[0, :] == output_vpu)



@pytest.mark.parametrize("filter_type", ["lowpass",
                                         "highpass"])
@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("f", [20, 1000, 20000])
@pytest.mark.parametrize("order", [4, 16])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
@pytest.mark.parametrize("q_format", [27, 31])
def test_nth_order_frame(filter_type, fs, f, order, n_chans, q_format):
    f = np.min([f, fs/2*0.95])
    if filter_type == "lowpass":
        if f < 50:
            return
        if f <= 100 and fs > 100000 and order >= 8:
            return
        filter = cbq.butterworth_lowpass(fs, n_chans, order, f, Q_sig=q_format)

    elif filter_type == "highpass":
        if f > 10000:
            return
        if f < 50 and fs > 100000:
            f = 30
        filter = cbq.butterworth_highpass(fs, n_chans, order, f, Q_sig=q_format)

    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)
    output_vpu = np.zeros_like(signal)
    frame_size = 1
    for n in range(len(signal_frames)):
        output_int[:, n*frame_size:(n+1)*frame_size] = filter.process_frame_int(signal_frames[n])
    assert np.all(output_int[0, :] == output_int)
    filter.reset_state()

    for n in range(len(signal_frames)):
        output_flt[:, n*frame_size:(n+1)*frame_size] = filter.process_frame(signal_frames[n])
    assert np.all(output_flt[0, :] == output_flt)
    filter.reset_state()

    for n in range(len(signal_frames)):
        output_vpu[:, n*frame_size:(n+1)*frame_size] = filter.process_frame_xcore(signal_frames[n])
    assert np.all(output_vpu[0, :] == output_vpu)


if __name__ == "__main__":
    test_peq_saturation(48000)
    # test_nth_butterworth("highpass", 20, 6, 16000)
