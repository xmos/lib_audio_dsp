# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.graphic_eq as geq
from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.signal_gen as gen
import pytest
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
  run_cmd = app + " " + str(bin_dir / "graphic_eq_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  print("run msg:\n", stdout.decode())

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt: geq.graphic_eq_10_band, sig_fl):
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
  all_filt_info = np.append(all_filt_info, np.array(filt.gains_int, dtype=np.int32))
  all_filt_info.tofile(test_dir / "gains.bin")

  out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav(test_dir)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()


@pytest.mark.parametrize("gains", [[-6, 0, -5, 1, -4, 2, -3, 3, -2, 4]])
def test_geq_c(in_signal, gains):

  peq = geq.graphic_eq_10_band (46050, 1, gains)

  filter_name = f"geq_{gains[0]}"
  single_test(peq, filter_name, in_signal)



if __name__ =="__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  
  test_geq_c(sig_fl, [0, -2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,])
