import pytest
import numpy as np
import audio_dsp.dsp.biquad as bq


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
    bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q, gain), b_shift=-2)


@pytest.mark.parametrize("filter_type", ["biquad_lowpass",
                                         "biquad_highpass",
                                         "biquad_notch",
                                         "biquad_allpass"])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_xpass_filters(filter_type, f, q, fs):
    filter_handle = getattr(bq, "make_%s" % filter_type)
    bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q), b_shift=0)


@pytest.mark.parametrize("filter_type", ["biquad_bandpass",
                                         "biquad_bandstop",])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_bandx_filters(filter_type, f, q, fs):
    filter_handle = getattr(bq, "make_%s" % filter_type)
    f = np.min([f, fs/2*0.95])
    high_q_stability_limit = 0.85
    if q >= 5 and f/(fs/2) > high_q_stability_limit:
        f = high_q_stability_limit*fs/2
    bq.biquad(filter_handle(fs, f, q), b_shift=0)


@pytest.mark.parametrize("f0", [20, 50, 100, 200, 500])
@pytest.mark.parametrize("fp", [20, 50, 100, 200, 500])
@pytest.mark.parametrize("q0", [0.5, 0.707, 1, 1.41, 2])
@pytest.mark.parametrize("qp", [0.5, 0.707, 1, 1.41, 2])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_linkwitz_filters(f0, fp, q0, qp, fs):
    bq.biquad(bq.make_biquad_linkwitz(fs, f0, q0, fp, qp), b_shift=0)
