# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.drc.drc_utils as drcu
import audio_dsp.dsp.signal_chain as sc
from audio_dsp.dsp.generic import Q_SIG, HEADROOM_DB


@pytest.mark.parametrize("ratio, warning", [(0, UserWarning),
                                            (0.9, UserWarning),
                                            (1, None),
                                            (5, None),
                                            (10000, None),
                                            (999999999999999999999999999, None)])
def test_compressor_ratio_helper(ratio, warning):
    if warning:
        with pytest.warns(warning):
            slope, slope_f32 = drcu.rms_compressor_slope_from_ratio(ratio)
    else:
            slope, slope_f32 = drcu.rms_compressor_slope_from_ratio(ratio)


@pytest.mark.parametrize("ratio, warning", [(0, UserWarning),
                                            (0.9, UserWarning),
                                            (1, None),
                                            (5, None),
                                            (10000, None),
                                            (999999999999999999999999999, None)])
def test_expander_ratio_helper(ratio, warning):
    if warning:
        with pytest.warns(warning):
            slope, slope_f32 = drcu.peak_expander_slope_from_ratio(ratio)
    else:
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


@pytest.mark.parametrize("time, warning", [(-1, UserWarning),
                                           (0, UserWarning),
                                           (1/48000, UserWarning),
                                           (3/48000, None),
                                           (1, None),
                                           (1000, None),
                                           ((4/48000)*(2**31), UserWarning),
                                           ])
def test_calc_alpha(time, warning):
    if warning:
        with pytest.warns(warning):
            alpha, alpha_int = drcu.alpha_from_time(time, 48000)
    else:
        alpha, alpha_int = drcu.alpha_from_time(time, 48000)


@pytest.mark.parametrize("gain_db, warning", [(-2000, None),
                                                   (0, None),
                                                   (25, UserWarning)])
def test_db_gain(gain_db, warning):
    if warning:
        with pytest.warns(warning):
            gain, gain_int = sc.db_to_qgain(gain_db)
    else:
            gain, gain_int = sc.db_to_qgain(gain_db)


@pytest.mark.parametrize("time, units, warning", [[10, "samples", None],
                                                  [128, "samples", None],
                                                  [1.7, "ms", None],
                                                  [0.94, "s", None],
                                                  [2, "s", None],
                                                  [-2, "s", UserWarning],
                                                  [2, "seconds", UserWarning]
                                                  ])
def test_time_to_samples(time, units, warning):
    if warning:
        with pytest.warns(warning):
            utils.time_to_samples(48000, time, units)
    else:
        utils.time_to_samples(48000, time, units)


if __name__ == "__main__":
    test_peak_threshold(25)