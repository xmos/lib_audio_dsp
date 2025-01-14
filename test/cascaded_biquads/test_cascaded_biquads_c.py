# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.cascaded_biquads as casc_bq
from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.signal_gen as gen
import pytest
import random
from ..test_utils import xdist_safe_bin_write

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000


def float_to_qxx(arr_float, q = Q_SIG, dtype = np.int32):
  arr_int32 = np.clip((np.array(arr_float) * (2**q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
  return arr_int32


def qxx_to_float(arr_int, q = Q_SIG):
  arr_float = np.array(arr_int).astype(np.float64) * (2 ** (-q))
  return arr_float


def get_sig(len=0.05):

  sig_fl = gen.log_chirp(fs, len, 0.5)
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
  run_cmd = app + " " + str(bin_dir / "cascaded_biquads_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  #print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt: casc_bq.cascaded_biquads_8, sig_fl):
  out_int = np.zeros(sig_fl.size)
  out_fl = np.zeros(sig_fl.size)
  
  for n in range(sig_fl.size):
    out_int[n] = filt.process_xcore(sig_fl[n])

  # sf.write(gen_dir / "sig_py_int.wav", out_int, fs, "PCM_24")

  return out_int


def single_test(filt, tname, sig_fl):
  test_dir = bin_dir / tname
  test_dir.mkdir(exist_ok = True, parents = True)

  all_filt_info = np.empty(0, dtype=np.int32)
  for biquad in filt.biquads:
    coeffs_arr = np.array(biquad.int_coeffs, dtype=np.int32)
    shift_arr = np.array(biquad.b_shift, dtype=np.int32)
    filt_info = np.append(coeffs_arr, shift_arr)
    all_filt_info = np.append(all_filt_info, filt_info)

  all_filt_info.tofile(test_dir / "coeffs.bin")

  out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav(test_dir)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()


@pytest.mark.parametrize("n_filters", [1, 5, 8])
@pytest.mark.parametrize("seed", [1, 5, 11])
def test_peq_c(in_signal, n_filters, seed):

  filter_spec = [['lowpass', fs * 0.4, 0.707],
                   ['highpass', fs * 0.001, 1],
                   ['peaking', 1000, 5, 10],
                   ['constant_q', 500, 1, -10],
                   ['notch', 2000, 1],
                   ['lowshelf', 200, 1, 3],
                   ['highshelf', 5000, 1, -2],
                   ['bypass'],
                   ['gain', -2]]
  random.Random(seed**n_filters*int(fs/1000)).shuffle(filter_spec)
  filter_spec = filter_spec[:n_filters]
  peq = casc_bq.parametric_eq_8band(fs, 1, filter_spec)

  filter_name = f"peq_{n_filters}_{seed}"
  single_test(peq, filter_name, in_signal)


@pytest.mark.parametrize("filter_type", ["lowpass",
                                         "highpass",])
@pytest.mark.parametrize("N", [4, 16]) # can only be 16 for now
@pytest.mark.parametrize("f", [20, 20000])
def test_nth_butterworth_c(in_signal, filter_type, N, f):
  f = np.min([f, fs / 2 * 0.95])
  filter_handle = getattr(casc_bq, f"butterworth_{filter_type}")

  filt = filter_handle(fs, 1, N, f)
  filter_name = f"butterworth_{filter_type}_{N}_{f}"
  single_test(filt, filter_name, in_signal)


if __name__ =="__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  
  #test_peq_c(sig_fl)
  test_nth_butterworth_c(sig_fl, "lowpass", 16, 2000)
