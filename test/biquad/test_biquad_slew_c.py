# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.biquad as bq
from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.signal_gen as gen
import pytest
from ..test_utils import xdist_safe_bin_write
from .test_biquad_c import float_to_qxx, qxx_to_float, in_signal

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000

def get_c_slew_wav(dir_name, sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / "biquad_slew_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  #print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py_slew(filt: bq.biquad_slew, sig_fl, coeffs_2):
    out_int = np.zeros(sig_fl.size)
  
    for n in range(sig_fl.size//2):
        out_int[n] = filt.process_xcore(sig_fl[n])

    filt.update_coeffs(coeffs_2)

    for n in range(sig_fl.size//2, sig_fl.size):
        out_int[n] = filt.process_xcore(sig_fl[n])

    sf.write(gen_dir / "sig_py_int.wav", out_int, fs, "PCM_24")

    return out_int


def single_slew_test(filt, tname, sig_fl, coeffs_2):
  test_dir = bin_dir / tname
  test_dir.mkdir(exist_ok = True, parents = True)
  coeffs_arr = np.array(filt.int_coeffs, dtype=np.int32)
  shift_arr = np.array(filt.b_shift, dtype=np.int32)
  filt_info = np.append(coeffs_arr, shift_arr)
  filt_info.tofile(test_dir / "coeffs.bin")

  _, coeffs_2_int = bq._round_and_check(coeffs_2, filt.b_shift)

  coeffs_2_arr = np.array(coeffs_2_int, dtype=np.int32)
  slew_arr = np.array(filt.slew_shift, dtype=np.int32)
  filt_2_info = np.append(coeffs_2_arr, slew_arr)
  filt_2_info.tofile(test_dir / "coeffs_2.bin")

  out_py_int = run_py_slew(filt, sig_fl, coeffs_2)
  out_c = get_c_slew_wav(test_dir)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)

@pytest.mark.parametrize("filter_1", [["biquad_constant_q", 100, 8, -10]])
@pytest.mark.parametrize("filter_2", [["biquad_constant_q", 10000, 8, -10]])
@pytest.mark.parametrize("slew_shift", [6])
def test_slew_c(in_signal, filter_1, filter_2, slew_shift):

  filter_type_1 = filter_1[0]
  filter_type_2 = filter_2[0]

  coeffs_hand_1 = getattr(bq, f"make_{filter_type_1}")
  coeffs_1 = coeffs_hand_1(fs, *filter_1[1:])
  coeffs_hand_2 = getattr(bq, f"make_{filter_type_2}")
  coeffs_2 = coeffs_hand_2(fs, *filter_2[1:])

  filt =bq.biquad_slew(coeffs_1, fs, 1, slew_shift=slew_shift)
  filter_name = f"slew_{filter_type_1}_{filter_1[1]}_{filter_type_2}_{filter_1[1]}"
  single_slew_test(filt, filter_name, in_signal, coeffs_2)

if __name__ == "__main__":
  from test_biquad_c import get_sig
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  test_slew_c(sig_fl, ["biquad_constant_q", 100, 8, -10], ["biquad_constant_q", 10000, 8, -10])