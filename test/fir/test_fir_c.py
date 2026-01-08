# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.fir as fir
from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.signal_gen as gen
import pytest
import os 
from test.test_utils import xdist_safe_bin_write, float_to_qxx, qxx_to_float, q_convert_flt


bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000


def get_sig(len=0.05):

  sig_fl = gen.log_chirp(fs, len, 0.5)
  sig_fl = q_convert_flt(sig_fl, 23, Q_SIG)
  sig_int = float_to_qxx(sig_fl)

  name = "sig_48k"
  sig_path = bin_dir /  str(name + ".bin")

  xdist_safe_bin_write(sig_int, sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl, int(fs), "PCM_24")
    
  return sig_fl


def get_c_wav(dir_name, sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / "fir_direct_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  #print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt: fir.fir_direct, sig_fl):
  out_int = np.zeros(sig_fl.size)
  
  for n in range(sig_fl.size):
    out_int[n] = filt.process_xcore(sig_fl[n])

  # sf.write(gen_dir / "sig_py_int.wav", out_int, fs, "PCM_24")

  return out_int


def single_test(filt, tname, sig_fl):
  test_dir = bin_dir / tname
  test_dir.mkdir(exist_ok = True, parents = True)
  coeffs_arr = np.array(filt.coeffs_int, dtype=np.int32,ndmin=1)
  taps_arr = np.array(filt.n_taps, dtype=np.int32, ndmin=1)
  shift_arr = np.array(filt.shift, dtype=np.int32, ndmin=1)
  filt_info = np.concatenate((shift_arr, taps_arr, coeffs_arr))
  filt_info.tofile(test_dir / "coeffs.bin")

  out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav(test_dir)
  shutil.rmtree(test_dir)

  overflow_samples = (np.abs(out_py_int) >= (2**(31-filt.Q_sig) - 1))
  np.testing.assert_allclose(out_c[~overflow_samples], out_py_int[~overflow_samples], rtol=0, atol=0)


@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()


# Note the filter coeffs files are defined in test/fir/conftest.py
@pytest.mark.parametrize("coeff_path", ["passthrough_filter.txt",
                                        "descending_coeffs.txt",
                                        "simple_low_pass.txt",
                                        "aggressive_high_pass.txt",
                                        "comb.txt",
                                        "tilt.txt"])
def test_fir_direct_c(in_signal, coeff_path):
  # this test compares the Python process_xcore fir implementation
  # against the lib_xcore_math version and checks for bit exactness

  filt = fir.fir_direct(fs, 1, Path(gen_dir, coeff_path))
  filter_name = f"fir_direct_{os.path.splitext(coeff_path)[0]}"
  single_test(filt, filter_name, in_signal)

if __name__ =="__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  test_fir_direct_c(sig_fl, "descending_coeffs.txt")
