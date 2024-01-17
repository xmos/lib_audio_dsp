import pytest
import numpy as np
import audio_dsp.dsp.biquad as bq
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils


def chirp_filter_test(filter: bq.biquad, fs):
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


@pytest.mark.parametrize("filter_type", ["biquad_peaking",
                                         "biquad_constant_q",
                                         "biquad_lowshelf",
                                         "biquad_highshelf",])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("gain", [-12, -10, -5, 0, 5, 10, 12])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_high_gain(filter_type, f, q, gain, fs):
    filter_handle = getattr(bq, "make_%s" % filter_type)
    filter = bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q, gain), fs, b_shift=2)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("filter_type", ["biquad_lowpass",
                                         "biquad_highpass",
                                         "biquad_notch",
                                         "biquad_allpass"])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_xpass_filters(filter_type, f, q, fs):

    if f < fs*5e-4 and filter_type == "biquad_lowpass":
        f = max(fs*5e-4, f)

    filter_handle = getattr(bq, "make_%s" % filter_type)
    filter = bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q), fs, b_shift=0)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("filter_type", ["biquad_bandpass",
                                         "biquad_bandstop",])
@pytest.mark.parametrize("f", [100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
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


@pytest.mark.parametrize("f0,", [20, 50, 100, 200, 500])
@pytest.mark.parametrize("fp_ratio", [0.4, 0.5, 1, 2, 4])
@pytest.mark.parametrize("q0", [0.5, 0.707, 1, 1.41, 2])
@pytest.mark.parametrize("qp", [0.5, 0.707, 1, 1.41, 2])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_linkwitz_filters(f0, fp_ratio, q0, qp, fs):

    if fs > 100000 and f0 < 50 and fp_ratio < 1:
        f0 = 30

    filter = bq.biquad_linkwitz(fs, f0, q0, f0*fp_ratio, qp)
    chirp_filter_test(filter, fs)


# TODO check biquad actually filters
# TODO check parameter generation
# TODO check sample rates - use f/fs


if __name__ == "__main__":
    # test_linkwitz_filters(500, 2, 20, 0.5, 48000)
    test_bandx_filters("biquad_bandstop", 10000, 10, 16000)
