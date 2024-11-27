
# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest
from pathlib import Path

import audio_dsp.dsp.fir as fir
import audio_dsp.dsp.td_block_fir as tbf
import audio_dsp.dsp.fd_block_fir as fbf
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
    fir_d = fir.fir_direct(48000, n_chans, Path(gen_dir, coeff_path))
    fir_btd = tbf.fir_block_td(48000, n_chans, Path(gen_dir, coeff_path), "dut",
    gen_dir, td_block_length=block_size)

    fir_bfd = fbf.fir_block_fd(48000, n_chans, Path(gen_dir, coeff_path), "dut",
    gen_dir, block_size, 0, 256)

    np.random.seed(0)
    # signal = sg.pink_noise(48000, 0.1, 0.5)
    signal = np.zeros(56)
    signal[:10] = 1
    signal[30:40] = 1
    signal = np.tile(signal, [n_chans, 1])
    signal[0] = -signal[0]
    frame_size = block_size

    signal_frames = utils.frame_signal(signal, frame_size, frame_size)

    out_flt_d = np.zeros_like(signal)
    out_flt_btd = np.zeros_like(signal)
    out_flt_bfd = np.zeros_like(signal)

    out_int = np.zeros_like(out_flt_d)

    for n in range(len(signal_frames)):
        out_flt_d  [:, n*frame_size:(n+1)*frame_size] = fir_d.process_frame(signal_frames[n])
        out_flt_btd[:, n*frame_size:(n+1)*frame_size] = fir_btd.process_frame(signal_frames[n])
        out_flt_bfd[:, n*frame_size:(n+1)*frame_size] = fir_bfd.process_frame(signal_frames[n])

    assert np.all(-out_flt_d[0, :] == out_flt_d[1:, :])
    np.testing.assert_allclose(out_flt_d, out_flt_btd, atol=2**-56, rtol=2**-42)
    np.testing.assert_allclose(out_flt_d, out_flt_bfd, atol=2**-56, rtol=2**-42)

    fir_d.reset_state()
    fir_btd.reset_state()

    for n in range(len(signal_frames)):
        out_int[:, n*frame_size:(n+1)*frame_size] = fir_d.process_frame_xcore(signal_frames[n])

    for n in range(1, n_chans):
        # rounding differences can occur between positive and negative signal
        np.testing.assert_allclose(-out_int[0, :], out_int[n, :], atol=(2**(-fir_d.Q_sig + 1)))

if __name__ == "__main__":
    test_frames("simple_low_pass.txt", 2, 8)