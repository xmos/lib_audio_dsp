import pytest

import audio_dsp.dsp.drc.drc_utils as drcu
from audio_dsp.dsp.generic import Q_SIG, HEADROOM_DB


@pytest.mark.parametrize("ratio", [pytest.param(0, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
                                   pytest.param(0.9, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
                                   1,
                                   5,
                                   10000,
                                   999999999999999999999999999])
def test_compressor_ratio_helper(ratio):
    slope, slope_f32 = drcu.rms_compressor_slope_from_ratio(ratio)


@pytest.mark.parametrize("ratio", [pytest.param(0, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
                                   pytest.param(0.9, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
                                   1,
                                   5,
                                   10000,
                                   999999999999999999999999999])
def test_expander_ratio_helper(ratio):
    slope, slope_f32 = drcu.peak_expander_slope_from_ratio(ratio)


@pytest.mark.parametrize("threshold_db, warning", [(-2000, None),
                                                   (0, None),
                                                   (HEADROOM_DB/2 + 1, UserWarning)])
def test_rms_threshold(threshold_db, warning):
    if warning:
        with pytest.warns(warning):
            thresh, thresh_int = drcu.calculate_threshold(threshold_db, Q_SIG, power=True)
    else:
            thresh, thresh_int = drcu.calculate_threshold(threshold_db, Q_SIG, power=True)


@pytest.mark.parametrize("threshold_db, warning", [(-2000, None),
                                                   (0, None),
                                                   (HEADROOM_DB + 1, UserWarning)])
def test_peak_threshold(threshold_db, warning):
    if warning:
        with pytest.warns(warning):
            thresh, thresh_int = drcu.calculate_threshold(threshold_db, Q_SIG, power=False)
    else:
            thresh, thresh_int = drcu.calculate_threshold(threshold_db, Q_SIG, power=False)


@pytest.mark.parametrize("time, warning", [pytest.param(-1, None, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
                                           (0, UserWarning),
                                           (1/48000, UserWarning),
                                           (3/48000, None),
                                           (1, None),
                                           (1000, None),
                                           pytest.param((4/48000)*(2**31), None, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
                                           ])
def test_calc_alpha(time, warning):
    if warning:
        with pytest.warns(warning):
            alpha, alpha_int = drcu.alpha_from_time(time, 48000)
    else:
        alpha, alpha_int = drcu.alpha_from_time(time, 48000)


if __name__ == "__main__":
    test_peak_threshold(25)