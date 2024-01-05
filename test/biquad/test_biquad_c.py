from scipy import signal
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import os
import audio_dsp.dsp.biquad as bq

build_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000
Q_format = 23

def float_to_qxx(arr_float, q = Q_format, dtype = np.int32):
  arr_int32 = np.clip((np.array(arr_float) * (2**q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
  return arr_int32

def qxx_to_float(arr_int, q = Q_format):
  arr_float = np.array(arr_int).astype(np.float64) * (2 ** (-q))
  return arr_float

def get_sig(f = 997, len = 0.25):
  time = np.arange(0, len, 1/fs)

  #sig_fl = 0.9 * np.sin(2 * np.pi * f * time)
  #sig_fl = 0.9 * signal.sawtooth(2 * np.pi * f * time)
  sig_fl = 0.5 * signal.chirp(time, 20, time[-1], 0.8 * fs / 2, "log", phi = -90)
  sig_int = float_to_qxx(sig_fl)
  name = "sig_48k"
  sig_int.tofile(build_dir /  str(name + ".bin"))
  sf.write(gen_dir / str(name + ".wav"), sig_fl, int(fs), "PCM_24")

  return sig_fl

def get_c_wav(sim = True):
  app = "xsim" if sim else "xrun"
  run_cmd = app + " " + str(build_dir / "biquad_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = build_dir, shell = True)
  print("run msg:\n", stdout)

  sig_bin = build_dir / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)

  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl

def run_py(sig_fl):
  filt = bq.biquad_lowpass(fs, 10000, 0.7)
  #filt = bq.biquad_peaking(fs, 1000, 1, 6)
  out_int = np.zeros(sig_fl.size)
  out_fl = np.zeros(sig_fl.size)
  
  coeff_copy = np.array(filt.coeffs)
  coeff_copy = (coeff_copy * 2**30).astype(np.int32)
  filt_info = np.append(coeff_copy.astype(np.int32), -filt.b_shift).astype(np.int32)
  filt_info.tofile(build_dir / "coeffs.bin")

  for n in range(sig_fl.size):
    #out_int[n] = filt.process_int(sig_fl[n])
    out_fl[n] = filt.process(sig_fl[n])

  sf.write(gen_dir / "sig_py.wav", out_fl, fs, "PCM_24")
  return out_fl, out_int

def test_biquad_c():
  build_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  out_py_fl, out_py_int = run_py(sig_fl)
  out_c = get_c_wav()

  #np.testing.assert_allclose(out_c, out_py_int, rtol=1e-5, verbose=True)
  np.testing.assert_allclose(out_c, out_py_fl, rtol=1e-4, verbose=True)
