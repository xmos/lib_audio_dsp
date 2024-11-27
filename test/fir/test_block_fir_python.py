
# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest
from pathlib import Path

import audio_dsp.dsp.fir as fir
import audio_dsp.dsp.td_block_fir as tbf
import audio_dsp.dsp.signal_gen as sg
import audio_dsp.dsp.utils as utils

gen_dir = Path(__file__).parent / "autogen"

# Note the filter coeffs files are defined in test/fir/conftest.py
@pytest.mark.parametrize("coeff_path", ["passthrough_filter.txt",
                                        "descending_coeffs.txt",
                                        "simple_low_pass.txt"])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
@pytest.mark.parametrize("block_size", [8])
def test_frames(coeff_path, n_chans, block_size):
    fut = fir.fir_direct(48000, n_chans, Path(gen_dir, coeff_path))
    fut2 = tbf.fir_block_td(48000, n_chans, Path(gen_dir, coeff_path), "dut",
    gen_dir, td_block_length=block_size)

    signal = sg.pink_noise(48000, 0.1, 0.5)
    signal = np.tile(signal, [n_chans, 1])
    signal[0] = -signal[0]
    frame_size = block_size

    signal_frames = utils.frame_signal(signal, frame_size, 1)

    out_flt = np.zeros_like(signal)
    out_flt2 = np.zeros_like(signal)

    out_int = np.zeros_like(out_flt)

    for n in range(len(signal_frames)):
        out_flt[:, n:n+frame_size] = fut.process_frame(signal_frames[n])
        out_flt2[:, n:n+frame_size] = fut2.process_frame(signal_frames[n])

    assert np.all(-out_flt[0, :] == out_flt[1:, :])
    np.testing.assert_allclose(out_flt, out_flt2, atol=2**-56, rtol=2**-42)

    fut.reset_state()
    fut2.reset_state()

    for n in range(len(signal_frames)):
        out_int[:, n:n+frame_size] = fut.process_frame_xcore(signal_frames[n])

    for n in range(1, n_chans):
        # rounding differences can occur between positive and negative signal
        np.testing.assert_allclose(-out_int[0, :], out_int[n, :], atol=(2**(-fut.Q_sig + 1)))

if __name__ == "__main__":
    test_frames("simple_low_pass.txt", 2, 8)