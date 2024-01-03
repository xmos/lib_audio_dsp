from scipy import signal
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import os
import audio_dsp.biquad as bq

build_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"
xcommon_path = os.environ.get("XMOS_CMAKE_PATH")

fs = 48000
Q_format = 23

def float_to_qxx(arr_float, q = Q_format, dtype = np.int32):
  arr_int32 = np.clip((np.array(arr_float) * (2**q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
  return arr_int32

def qxx_to_float(arr_int, q = Q_format):
  arr_float = np.array(arr_int).astype(np.float64) * (2 ** (-q))
  return arr_float

def get_sig(f = 1000, len = 0.25):
  time = np.arange(0, len, 1/fs)

  #sig_fl = 0.9 * np.sin(2 * np.pi * f * time)
  #sig_fl = 0.9 * signal.sawtooth(2 * np.pi * f * time)
  sig_fl = 0.25 * signal.chirp(time, 20, time[-1], 0.8 * fs / 2, "log", phi = -90)
  sig_int = float_to_qxx(sig_fl)
  name = "sig_48k"
  sig_int.tofile(build_dir /  str(name + ".bin"))
  sf.write(gen_dir / str(name + ".wav"), sig_fl, int(fs), "PCM_24")

  return sig_fl, sig_int

def get_c_wav(build = False, sim = True):
  if build and xcommon_path != None:
    make_cmd = f"XMOS_CMAKE_PATH={xcommon_path} make -C " + str(build_dir)
    stdout = subprocess.check_output(make_cmd, cwd=build_dir, shell=True)
    print("build msg:\n", stdout, "\n")
  elif build and xcommon_path == None:
    assert 0, "XMOS_CMAKE_PATH is not set"

  if sim: app = "xsim"
  else: app = "xrun"
  run_cmd = app + " " + str(build_dir / "dummy_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = build_dir, shell = True)
  print("run msg:\n", stdout)

  sig_bin = build_dir / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  #sf.write(gen_dir / "sig_c.wav", sig_int, fs, "PCM_32")
  sf.write(gen_dir / "sig_c.wav", sig_int<<8, fs, "PCM_24")

def run_py(sig_fl):
  filt = bq.biquad_lowpass(fs, 1000, 0.7)
  #filt = bq.biquad_peaking(fs, 1000, 1, 6)
  output = np.zeros(sig_fl.size)
  coeff_copy = np.array(filt.coeffs)
  coeff_copy = (coeff_copy * 2**30).astype(np.int32)
  print(coeff_copy)
  for n in range(sig_fl.size):
    output[n] = filt.process_int(sig_fl[n])
  sf.write(gen_dir / "sig_py.wav", output, fs, "PCM_24")

sig_fl, sig_int = get_sig()
get_c_wav()
run_py(sig_fl)
