# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest
import numpy as np
import audio_dsp.dsp.biquad as bq
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils


def saturation_test(filter: bq.biquad, fs):

    signal = 2**(np.arange(0, 31.5, 0.5)) - 1
    signal = np.repeat(signal, 2)
    signal[::2] *= -1

    # # used for lib_xcore_math biquad test
    # sigint = (np.round(signal).astype(np.int32))
    # np.savetxt("sig.csv", sigint, fmt="%i", delimiter=",")
    signal = 2.0**30 - 1
    signal = np.repeat(signal, 4)
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


def chirp_filter_test(filter: bq.biquad, fs):
    length = 0.05
    signal = gen.log_chirp(fs, length, 1.0)

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

    # after saturation, the implementations diverge, but they should
    # initially saturate at the same sample
    first_sat = np.argmax(np.abs(output_flt) >= 0.9999999995343387)
    top_half[first_sat + 1:] = False

    if np.any(top_half):
        error_flt = np.abs(utils.db(output_int[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055
        error_vpu = np.abs(utils.db(output_int[top_half])-utils.db(output_vpu[top_half]))
        mean_error_vpu = utils.db(np.nanmean(utils.db2gain(error_vpu)))
        assert mean_error_vpu < 0.05


def test_4_coeff_overflow():
    fs = 48000
    filter = bq.biquad([1.0, -1.5, 0.5625, 1.5, -0.5625], fs, Q_sig=31)
    saturation_test(filter, 48000)


@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
@pytest.mark.parametrize("amplitude", [0.5, 1, 2, 16])
def test_bypass(fs, amplitude):
    filter = bq.biquad_bypass(fs, 1)
    length = 0.05
    signal = gen.log_chirp(fs, length, amplitude)
    signal = utils.saturate_float_array(signal, filter.Q_sig)


    output_int = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))
    output_xcore = np.zeros(len(signal))

    for n in np.arange(len(signal)):
        output_int[n] = filter.process_int(signal[n])
    filter.reset_state()
    for n in np.arange(len(signal)):
        output_flt[n] = filter.process(signal[n])
    filter.reset_state()
    for n in np.arange(len(signal)):
        output_xcore[n] = filter.process_xcore(signal[n])

    np.testing.assert_array_equal(signal, output_flt)
    np.testing.assert_allclose(signal, output_int, atol=2**-32)
    np.testing.assert_allclose(signal, output_xcore, atol=2**-32)


@pytest.mark.parametrize("filter_type", ["biquad_peaking",
                                         "biquad_constant_q"])
@pytest.mark.parametrize("f", [20, 100, 1000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 1, 2, 10])
@pytest.mark.parametrize("gain", [-12, -6, 0, 6, 12])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_peaking_filters(filter_type, f, q, gain, fs):
    if f < fs*5e-4:
        f = max(fs*5e-4, f)
    filter_handle = getattr(bq, "make_%s" % filter_type)
    filter = bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q, gain), fs, b_shift=bq.BOOST_BSHIFT)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("filter_type", ["biquad_lowshelf",
                                         "biquad_highshelf",])
@pytest.mark.parametrize("f", [20, 100, 1000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 1, 2])
@pytest.mark.parametrize("gain", [-12, -6, 0, 6, 12])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_shelf_filters(filter_type, f, q, gain, fs):

    if f < fs*5e-4:
        f = max(fs*5e-4, f)

    filter_handle = getattr(bq, "make_%s" % filter_type)
    filter = bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q, gain), fs, b_shift=bq.BOOST_BSHIFT)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("filter_type", ["biquad_lowpass",
                                         "biquad_highpass",
                                         "biquad_notch",
                                         "biquad_allpass"])
@pytest.mark.parametrize("f", [20, 100, 1000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 1, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_xpass_filters(filter_type, f, q, fs):

    if f < fs*5e-4 and filter_type == "biquad_lowpass":
        f = max(fs*5e-4, f)

    filter_handle = getattr(bq, "make_%s" % filter_type)
    filter = bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q), fs, b_shift=0)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("filter_type", ["biquad_bandpass",
                                         "biquad_bandstop",])
@pytest.mark.parametrize("f", [100, 1000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 1, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_bandx_filters(filter_type, f, q, fs):
    filter_handle = getattr(bq, "make_%s" % filter_type)
    f = np.min([f, fs/2*0.95])
    if f < fs*1e-3:
        q = max(0.5, q)
    high_q_stability_limit = 0.85
    if q >= 5 and f/(fs/2) > high_q_stability_limit:
        f = high_q_stability_limit*fs/2

    filter = bq.biquad(filter_handle(fs, f, q), fs, b_shift=0)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("f0,", [20, 50, 100, 200])
@pytest.mark.parametrize("fp_ratio", [0.4, 1, 4])
@pytest.mark.parametrize("q0, qp", [(0.5, 2),
                                    (2, 0.5),
                                    (0.707, 0.707)])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_linkwitz_filters(f0, fp_ratio, q0, qp, fs):

    if fs > 100000 and f0 < 50 and fp_ratio < 1:
        f0 = 30

    filter = bq.biquad_linkwitz(fs, 1, f0, q0, f0*fp_ratio, qp)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("gain", [-10, 0, 10])
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 88200, 96000, 192000])
def test_gain_filters(gain, fs):

    filter = bq.biquad_gain(fs, 1, gain)
    chirp_filter_test(filter, fs)

@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("filter_n", np.arange(9))
@pytest.mark.parametrize("n_chans", [1, 2, 4])
def test_frames(filter_n, fs, n_chans):
    filter_spec = [['lowpass', fs*0.4, 0.707],
                   ['highpass', fs*0.001, 1],
                   ['peaking', fs*1000/48000, 5, 10],
                   ['constant_q', fs*500/48000, 1, -10],
                   ['notch', fs*2000/48000, 1],
                   ['lowshelf', fs*200/48000, 1, 3],
                   ['highshelf', fs*5000/48000, 1, -2],
                   ['bypass'],
                   ['gain', -2]]

    filter_spec = filter_spec[filter_n]

    class_name = f"biquad_{filter_spec[0]}"
    class_handle = getattr(bq, class_name)
    filter = class_handle(fs, n_chans, *filter_spec[1:])

    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)
    output_vpu = np.zeros_like(signal)
    frame_size = 1
    for n in range(len(signal_frames)):
        output_int[:, n:n+frame_size] = filter.process_frame_int(signal_frames[n])
    assert np.all(output_int[0, :] == output_int)
    filter.reset_state()

    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = filter.process_frame(signal_frames[n])
    assert np.all(output_flt[0, :] == output_flt)
    filter.reset_state()

    for n in range(len(signal_frames)):
        output_vpu[:, n:n+frame_size] = filter.process_frame_xcore(signal_frames[n])
    assert np.all(output_vpu[0, :] == output_vpu)




# TODO check biquad actually filters
# TODO check parameter generation
# TODO check sample rates - use f/fs
# TODO add mute tests

if __name__ == "__main__":
    # test_linkwitz_filters(500, 2, 20, 0.5, 48000)
    # test_bandx_filters("biquad_bandstop", 10000, 10, 16000)
    # test_bypass(96000, 1)
    # test_gain_filters(5, 16000)
    test_peaking_filters("biquad_peaking", 20, 0.5, 12, 16000)
