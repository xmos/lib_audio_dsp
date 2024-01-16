import pytest
import numpy as np
import random
import scipy.signal as spsig

import audio_dsp.dsp.cascade_biquads as cbq
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils


def chirp_filter_test(filter: cbq.cascaded_biquads, fs):
    length = 0.5
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
        output_vpu[n] = filter.process_vpu(signal[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_int) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_int[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055
        error_vpu = np.abs(utils.db(output_int[top_half])-utils.db(output_vpu[top_half]))
        mean_error_vpu = utils.db(np.nanmean(utils.db2gain(error_vpu)))
        assert mean_error_vpu < 0.05


@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
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
                   ['bypass'],]
                #    ['gain', -2]]
    random.Random(seed**n_filters).shuffle(filter_spec)
    filter_spec = filter_spec[:n_filters]
    peq = cbq.parametric_eq(fs, filter_spec)
    chirp_filter_test(peq, fs)

# TODO higher order filter tests
    
@pytest.mark.parametrize("filter_type", ["lowpass",
                                         "highpass"])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("order", [4, 6, 8, 10, 12])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_nth_butterworth(filter_type, f, order, fs):
    f = np.min([f, fs/2*0.95])
    if filter_type == "lowpass":
        if f < 50:
            return
        filter = cbq.butterworth_lowpass(fs, order, f)
    elif filter_type == "highpass":
        if f > 10000:
            return
        if f < 50 and fs > 100000:
            f = 30
        filter = cbq.butterworth_highpass(fs, order, f)

    # chirp_filter_test(filter, fs)

    # compare against scipy butterworth response
    w, h = filter.freq_response(1024)
    b, a = spsig.butter(order, f, fs=fs, btype=filter_type[:-4])

    w_ref, h_ref = spsig.freqz(b, a, worN=1024)

    top_half = utils.db(h) > -50
    err = np.abs(utils.db(h[top_half]) - utils.db(h_ref[top_half]))
    mean_error = utils.db(np.nanmean(utils.db2gain(err)))

    assert mean_error < 0.1

    print(mean_error)
