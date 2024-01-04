import pytest
import numpy as np
import audio_dsp.dsp.biquad as bq


@pytest.mark.parametrize("filter_type", ["biquad_peaking",
                                         "biquad_constant_q",
                                         "biquad_lowshelf",
                                         "biquad_highshelf",])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("gain", [-15, -10, -5, 0, 5, 10, 15])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_high_gain(filter_type, f, q, gain, fs):
    filter_handle = getattr(bq, "make_%s" % filter_type)
    bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q, gain), b_shift=-3)


@pytest.mark.parametrize("filter_type", ["biquad_lowpass",
                                         "biquad_highpass",
                                         "biquad_bandpass",
                                         "biquad_bandstop",
                                         "biquad_notch",
                                         "biquad_allpass"])
@pytest.mark.parametrize("f", [20, 100, 200, 1000, 2000, 10000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.707, 1, 1.41, 2, 5, 10])
@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 88200, 96000, 192000])
def test_xpass_filters(filter_type, f, q, fs):
    filter_handle = getattr(bq, "make_%s" % filter_type)
    bq.biquad(filter_handle(fs, np.min([f, fs/2*0.95]), q), b_shift=0)
