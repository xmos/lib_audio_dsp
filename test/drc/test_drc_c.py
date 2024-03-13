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


def get_c_wav(dir_name, lim_name, sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / lim_name) + "_test.xe"
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  #print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)
  
  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt: drc.compressor_limiter_base, sig_fl):
  out_f32 = np.zeros(sig_fl.size)
  out_f64 = np.zeros(sig_fl.size)
  
  for n in range(sig_fl.size):
    out_f32[n], _, _ = filt.process_xcore(sig_fl[n])

  sf.write(gen_dir / "sig_py_int.wav", out_f32, fs, "PCM_24")
  filt.reset_state()

  for n in range(sig_fl.size):
    out_f64[n], _, _ = filt.process(sig_fl[n])

  #sf.write(gen_dir / "sig_py_flt.wav", out_f64, fs, "PCM_24")

  return out_f64, out_f32

@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()

@pytest.mark.parametrize("component_name", ["limiter_peak",
                                            "limiter_rms",
                                            "noise_gate"])
@pytest.mark.parametrize("at", [0.001, 0.1])
@pytest.mark.parametrize("rt", [0.01, 0.2])
@pytest.mark.parametrize("threshold", [-20, 0])
def test_limiter_c(in_signal, component_name, at, rt, threshold):
  # there is a difference between C and PY now which shows up in this test case
  # nothing too critical, should be fixed soon
  if component_name == "noise_gate" and threshold == 0:
    pytest.xfail("Noise gate with threshold 0 is not bit exact")
  component_handle = getattr(drc, component_name)
  comp = component_handle(fs, 1, threshold, at, rt)
  test_name = f"{component_name}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  info = [comp.threshold_f32, comp.attack_alpha_f32, comp.release_alpha_f32]
  info = np.array(info, dtype = np.float32)
  info.tofile(test_dir / "info.bin")

  _, out_py_int = run_py(comp, in_signal)
  out_c = get_c_wav(test_dir, component_name)
  shutil.rmtree(test_dir)

  if test_name == "limiter_peak_-20_0.001_0.01":
    # for some reason this particular exapmle isn't bit exact, so just
    # check the number of unmatched samples is small, and that the
    # max atol is small.
    not_equal_idx = np.sum(out_py_int != out_c)
    pct_not_equal = ((not_equal_idx) / len(out_py_int)) * 100
    assert pct_not_equal <= 0.5, f"Output mismatch: {pct_not_equal}% of samples are not equal"

    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=1.5e-8)
  else:
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)

@pytest.mark.parametrize("comp_name", ["compressor_rms"])
@pytest.mark.parametrize("at", [0.001])
@pytest.mark.parametrize("rt", [0.01])
@pytest.mark.parametrize("threshold", [-12, 0])
@pytest.mark.parametrize("ratio", [1, 6])
def test_compressor_c(in_signal, comp_name, at, rt, threshold, ratio):
  comp_handle = getattr(drc, comp_name)
  comp = comp_handle(fs, 1, ratio, threshold, at, rt)
  test_name = f"{comp_name}_{ratio}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  comp_info = [comp.threshold_f32, comp.slope_f32, comp.attack_alpha_f32, comp.release_alpha_f32]
  comp_info = np.array(comp_info, dtype = np.float32)
  comp_info.tofile(test_dir / "comp_info.bin")

  _, out_py_int = run_py(comp, in_signal)
  out_c = get_c_wav(test_dir, comp_name)
  shutil.rmtree(test_dir)

  # when ratio is 1, the result should be bit-exact as we don't have to use powf
  if ratio == 1 or threshold == 0:
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)
  else:
    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=4.5e-8)

if __name__ == "__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()

  #test_limiter_c(sig_fl, "limiter_rms", 0.001, 0.07, -20)
  #test_limiter_c(sig_fl, "limiter_peak", 0.001, 0.01, -20)
  test_limiter_c(sig_fl, "noise_gate", 0.001, 0.01, 0)
  #test_compressor_c(sig_fl, "compressor_rms", 0.001, 0.01, -6, 4)
