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


def run_py(filt, sig_fl):
  out_f32 = np.zeros(sig_fl.size)
  out_f64 = np.zeros(sig_fl.size)
  
  for n in range(sig_fl.size):
    out_f32[n], _, _ = filt.process_f32(sig_fl[n])

  sf.write(gen_dir / "sig_py_int.wav", out_f32, fs, "PCM_24")
  filt.reset_state()

  for n in range(sig_fl.size):
    out_f64[n], _, _ = filt.process(sig_fl[n])

  sf.write(gen_dir / "sig_py_flt.wav", out_f64, fs, "PCM_24")

  return out_f64, out_f32


def single_test(lim, lim_name, tname, sig_fl):
  test_dir = bin_dir / tname
  test_dir.mkdir(exist_ok = True, parents = True)

  lim_info = [lim.threshold_f32, lim.attack_alpha_f32, lim.release_alpha_f32]
  lim_info = np.array(lim_info, dtype = np.float32)
  lim_info.tofile(test_dir / "lim_info.bin")

  out_py_fl, out_py_int = run_py(lim, sig_fl)
  out_c = get_c_wav(test_dir, lim_name)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=2e-8)

@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()

@pytest.mark.parametrize("lim_name", ["limiter_peak",
                                      "limiter_rms"])
@pytest.mark.parametrize("at", [0.001, 0.1])
@pytest.mark.parametrize("rt", [0.01, 0.2])
@pytest.mark.parametrize("threshold", [-20, 0])
def test_limiter_c(in_signal, lim_name, at, rt, threshold):
  lim_handle = getattr(drc, lim_name)
  lim = lim_handle(fs, 1, threshold, at, rt)
  test_name = f"{lim_name}_{threshold}_{at}_{rt}"

  single_test(lim, lim_name, test_name, in_signal)

if __name__ == "__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()

  #test_limiter_c(sig_fl, "limiter_rms", 0.001, 0.07, -20)
  test_limiter_c(sig_fl, "limiter_peak", 0.01, 0.07, -20)
