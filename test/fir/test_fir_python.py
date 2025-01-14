# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest
from pathlib import Path

import audio_dsp.dsp.fir as fir
import audio_dsp.dsp.signal_gen as sg
import audio_dsp.dsp.utils as utils

gen_dir = Path(__file__).parent / "autogen"

# Note the filter coeffs files are defined in test/fir/conftest.py
@pytest.mark.parametrize("coeff_path", ["passthrough_filter.txt",
                                        "descending_coeffs.txt",
                                        "simple_low_pass.txt",
                                        "aggressive_high_pass.txt",
                                        "comb.txt",
                                        "tilt.txt"])
def test_basic(coeff_path):
    fut = fir.fir_direct(48000, 1, Path(gen_dir, coeff_path))

    signal = sg.pink_noise(48000, 0.1, 0.5)
    # signal = np.zeros(1000)
    # signal[0] = 1

    coeffs = np.loadtxt(Path(gen_dir, coeff_path))
    out_ref = np.convolve(signal, coeffs)[:len(signal)]

    out_flt = np.zeros_like(signal)
    out_int = np.zeros_like(out_flt)

    for n in range(len(signal)):
        out_flt[n] = fut.process(signal[n])
    
    fut.reset_state()

    for n in range(len(signal)):
        out_int[n] = fut.process_xcore(signal[n])

    # difference in convolution implementations means flt and ref aren't
    # bit exact, especially after saturation!
    unsaturated = ((out_ref > float(-(2 ** (31 - fut.Q_sig)))) & 
                    (out_ref < float((2**31 - 1) / 2**fut.Q_sig)))
    np.testing.assert_allclose(out_flt[unsaturated], out_ref[unsaturated], atol=2**-52)

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(out_flt) > -100
    if np.any(top_half):
        error_flt = np.abs(utils.db(out_int[top_half])-utils.db(out_flt[top_half]))
        mean_error_int = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_int < 0.016
        np.testing.assert_allclose(out_flt, out_int, atol=2**(-21))


# Note the filter coeffs files are defined in test/fir/conftest.py
@pytest.mark.parametrize("coeff_path", ["passthrough_filter.txt",
                                        "descending_coeffs.txt",
                                        "simple_low_pass.txt"])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
def test_frames(coeff_path, n_chans):
    fut = fir.fir_direct(48000, n_chans, Path(gen_dir, coeff_path))

    signal = sg.pink_noise(48000, 0.1, 0.5)
    signal = np.tile(signal, [n_chans, 1])
    signal[0] = -signal[0]
    frame_size = 1

    signal_frames = utils.frame_signal(signal, frame_size, 1)

    out_flt = np.zeros_like(signal)
    out_int = np.zeros_like(out_flt)

    for n in range(len(signal_frames)):
        out_flt[:, n*frame_size:(n+1)*frame_size] = fut.process_frame(signal_frames[n])
    assert np.all(-out_flt[0, :] == out_flt[1:, :])
    fut.reset_state()

    for n in range(len(signal_frames)):
        out_int[:, n*frame_size:(n+1)*frame_size] = fut.process_frame_xcore(signal_frames[n])

    for n in range(1, n_chans):
        # rounding differences can occur between positive and negative signal
        np.testing.assert_allclose(-out_int[0, :], out_int[n, :], atol=(2**(-fut.Q_sig)))


if __name__ =="__main__":
    # test_basic("simple_low_pass.txt")
    # for n in range(100):
    test_frames("simple_low_pass.txt", 2)