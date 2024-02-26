# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.signal_chain as sc
from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.signal_gen as gen
import pytest
import random

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
  sig_fl = []
  sig_fl.append(gen.sin(fs, len, 997, 0.7))
  sig_fl.append(gen.sin(fs, len, 100, 0.7))
  sig_fl = np.stack(sig_fl, axis=0)
  sig_int = float_to_qxx(sig_fl)

  name = "sig_48k"
  sig_int[0].tofile(bin_dir /  f"{name}.bin")
  sf.write(gen_dir / f"{name}.wav", sig_fl[0], int(fs), "PCM_24")

  name = "sig1_48k"
  sig_int[1].tofile(bin_dir /  f"{name}.bin")
  sf.write(gen_dir / f"{name}.wav", sig_fl[1], int(fs), "PCM_24")

  return sig_fl

def get_c_wav(dir_name, comp_name, sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / f"{comp_name}_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  #print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl

def write_gain(test_dir, gain):
  all_filt_info = np.empty(0, dtype=np.int32)
  all_filt_info = np.append(all_filt_info, gain)
  all_filt_info.tofile(test_dir / "gain.bin")

def single_test(filt, test_dir, fname, sig_fl):
  out_py = np.zeros(sig_fl.shape[1])

  for n in range(sig_fl.shape[1]):
    out_py[n] = filt.process_xcore(sig_fl[:, n])
  
  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, fname)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)

@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()

@pytest.mark.parametrize("gain_dB", [-10, 0, 24])
def test_gains_c(in_signal, gain_dB):
  filt = sc.fixed_gain(fs, 1, gain_dB)
  test_dir = bin_dir / f"fixed_gain_{gain_dB}"
  test_dir.mkdir(exist_ok = True, parents = True)
  write_gain(test_dir, filt.gain_int)

  out_py = np.zeros(in_signal.shape[1])
  
  for n in range(in_signal.shape[1]):
    out_py[n] = filt.process_xcore(in_signal[0][n])

  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, "fixed_gain")
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)

def test_subtractor_c(in_signal):
  filt = sc.subtractor(fs)
  test_dir = bin_dir / "subtractor"
  test_dir.mkdir(exist_ok = True, parents = True)

  single_test(filt, test_dir, "subtractor", in_signal)

def test_adder_c(in_signal):
  filt = sc.adder(fs, 2)
  test_dir = bin_dir / "adder"
  test_dir.mkdir(exist_ok = True, parents = True)

  single_test(filt, test_dir, "adder", in_signal)

@pytest.mark.parametrize("gain_dB", [-12, -6, 0])
def test_mixer_c(in_signal, gain_dB):
  filt = sc.mixer(fs, 2, gain_dB)
  test_dir = bin_dir / f"mixer_{gain_dB}"
  test_dir.mkdir(exist_ok = True, parents = True)
  write_gain(test_dir, filt.gain_int)

  single_test(filt, test_dir, "mixer", in_signal)

if __name__ =="__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  
  #test_gains_c(sig_fl, -6)
  test_subtractor_c(sig_fl)
  test_adder_c(sig_fl)
  test_mixer_c(sig_fl, -3)
