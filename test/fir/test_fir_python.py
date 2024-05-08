# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest
from pathlib import Path

import audio_dsp.dsp.fir as fir
import audio_dsp.dsp.signal_gen as sg

gen_dir = Path(__file__).parent / "autogen"

# Note the filter coeffs are defined in conftest
@pytest.mark.parametrize("coeff_path", ["passthrough_filter.txt",
                                        "descending_coeffs.txt",
                                        "simple_low_pass.txt"])
def test_basic(coeff_path):
    fut = fir.fir_direct(48000, 1, Path(gen_dir, coeff_path))


    signal = sg.pink_noise(48000, 1, 0.5)
    # signal = np.zeros(1000)
    # signal[0:2] = 1

    coeffs = np.loadtxt(coeff_path)
    out_ref = np.convolve(signal, coeffs)[:len(signal)]

    out_flt = np.zeros_like(signal)

    for n in range(len(signal)):
        out_flt[n] = fut.process(signal[n])

    # difference in convolution implementations means flt and ref aren't
    # bit exact
    np.testing.assert_allclose(out_flt, out_ref, atol=2**-52)


if __name__ =="__main__":
    test_basic("simple_low_pass.txt")