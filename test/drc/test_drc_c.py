# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.drc as drc
from audio_dsp.dsp.generic import Q_SIG
from audio_dsp.dsp.signal_gen import quantize_signal
import pytest

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
  time = np.arange(0, len, 1/fs)
  sig_fl = 0.8 * np.sin(2 * np.pi * 997 * time) * np.sin(2 * np.pi * 100 * time)
  sig_fl = quantize_signal(sig_fl, 24)
  sig_int = float_to_qxx(sig_fl)

  name = "sig_48k"
  sig_int.tofile(bin_dir /  str(name + ".bin"))
  sf.write(gen_dir / str(name + ".wav"), sig_fl, int(fs), "PCM_24")

  return sig_fl

def get_c_wav(dir_name, bin_name, verbose = False, sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / bin_name) + "_test.xe"
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  if verbose: print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)
  
  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt, sig_fl, single_output, run_f64 = False):
  out_xpy = np.zeros(sig_fl.size)
  out_pypy = np.zeros(sig_fl.size)
  
  if single_output:
    for n in range(sig_fl.size):
      out_xpy[n] = filt.process_xcore(sig_fl[n])
      if run_f64:
        filt.reset_state()
        out_pypy[n] = filt.process(sig_fl[n])
  else:
    for n in range(sig_fl.size):
      out_xpy[n] = filt.process_xcore(sig_fl[n])[0]
      if run_f64:
        filt.reset_state()
        out_pypy[n] = filt.process(sig_fl[n])[0]

  sf.write(gen_dir / "sig_py_int.wav", out_xpy, fs, "PCM_24")

  if run_f64:
    sf.write(gen_dir / "sig_py_flt.wav", out_pypy, fs, "PCM_24")
    return out_pypy, out_xpy
  else:
    return out_xpy

@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()

@pytest.mark.parametrize("env_name", ["envelope_detector_peak",
                                      "envelope_detector_rms"])
@pytest.mark.parametrize("at", [0.007, 0.15])
@pytest.mark.parametrize("rt", [0.005, 0.2])
def test_env_det_c(in_signal, env_name, at, rt):
  env_handle = getattr(drc, env_name)
  env = env_handle(fs, 1, at, rt)
  test_name = f"{env_name}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  env_info = [env.attack_alpha_int, env.release_alpha_int]
  env_info = np.array(env_info, dtype = np.int32)
  env_info.tofile(test_dir / "env_info.bin")

  out_py_int = run_py(env, in_signal, True)  
  out_c = get_c_wav(test_dir, env_name)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)

@pytest.mark.parametrize("component_name", ["limiter_peak",
                                            "limiter_rms",
                                            "hard_limiter_peak",
                                            "clipper",
                                            "noise_gate"
                                            ])
@pytest.mark.parametrize("at", [0.001, 0.1])
@pytest.mark.parametrize("rt", [0.01, 0.2])
@pytest.mark.parametrize("threshold", [-20, 0])
def test_limiter_c(in_signal, component_name, at, rt, threshold):
  # Skip the test as the clipper doesn't use attack and release times
  if component_name == "clipper" and (at == 0.1 or rt == 0.2):
    return

  component_handle = getattr(drc, component_name)
  if component_name == "clipper":
    comp = component_handle(fs, 1, threshold)
    test_name = f"{component_name}_{threshold}"
  else:
    comp = component_handle(fs, 1, threshold, at, rt)
    test_name = f"{component_name}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  if component_name == "clipper":
    info = [comp.threshold_int]
    single_output = True
  else:
    info = [comp.threshold_int, comp.attack_alpha_int, comp.release_alpha_int]
    single_output = False
  info = np.array(info, dtype = np.int32)
  info.tofile(test_dir / "info.bin")


  out_py_int = run_py(comp, in_signal, single_output)
  out_c = get_c_wav(test_dir, component_name)
  shutil.rmtree(test_dir)

  if component_name == "limiter_rms" and threshold != 0:
    # python uses float sqrt when C uses the fixed point one, so expect some diff
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=7.5e-9)
  else:
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)

@pytest.mark.parametrize("comp_name", ["compressor_rms",
                                       "noise_suppressor"])
@pytest.mark.parametrize("at", [0.005])
@pytest.mark.parametrize("rt", [0.120])
@pytest.mark.parametrize("threshold", [-12, 0])
@pytest.mark.parametrize("ratio", [1, 6])
def test_compressor_c(in_signal, comp_name, at, rt, threshold, ratio):
  # for the noise suppressor the lowest sensible threshold is -35
  if comp_name == "noise_suppressor" and threshold == -12:
    threshold = -35
  comp_handle = getattr(drc, comp_name)
  comp = comp_handle(fs, 1, ratio, threshold, at, rt)
  test_name = f"{comp_name}_{ratio}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  # numpy doesn't like to have an array with different types
  # so create separate arrays, cast to bytes, append, write
  info = [comp.threshold_int, comp.attack_alpha_int, comp.release_alpha_int]
  info = np.array(info, dtype=np.int32)
  info1 = np.array(comp.slope_f32, dtype=np.float32)
  info = info.tobytes()
  info1 = info1.tobytes()
  info = np.append(info, info1)
  info.tofile(test_dir / "info.bin")

  out_py_int = run_py(comp, in_signal, False)
  out_c = get_c_wav(test_dir, comp_name)
  shutil.rmtree(test_dir)

  # when ratio is 1, the result should be bit-exact as we don't have to use powf
  if ratio == 1 or (threshold == 0 and comp_name != "noise_suppressor"):
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)
  else:
    # tolerace is the 24b float32 mantissa
    tol = 2**(np.ceil(np.log2(np.max(out_c))) - 24)
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=tol)

if __name__ == "__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()

  test_env_det_c(sig_fl, "envelope_detector_rms", 0.001, 0.01)
  test_limiter_c(sig_fl, "limiter_rms", 0.001, 0.07, -10)
  test_compressor_c(sig_fl, "noise_suppressor", 0.001, 0.01, -1, 5)
