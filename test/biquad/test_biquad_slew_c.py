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
from .test_biquad_c import float_to_qxx, qxx_to_float
import os

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
        out_int[n] = filt.process_channels_xcore([sig_fl[n]])[0]

    filt.update_coeffs(coeffs_2)

    for n in range(sig_fl.size//2, sig_fl.size):
        out_int[n] = filt.process_channels_xcore([sig_fl[n]])[0]

    sf.write(gen_dir / "sig_py_int.wav", out_int, fs, "PCM_24")

    return out_int


def single_slew_test(filt, tname, sig_fl, filt_2, coeffs_2):
  test_dir = bin_dir / tname
  test_dir.mkdir(exist_ok = True, parents = True)
  coeffs_arr = np.array(filt.int_coeffs, dtype=np.int32)
  shift_arr = np.array(filt.b_shift, dtype=np.int32)
  slew_arr = np.array(filt.slew_shift, dtype=np.int32)
  filt_info = np.append(coeffs_arr, (shift_arr, slew_arr))

  filt_info.tofile(test_dir / "coeffs.bin")

  coeffs_arr = np.array(filt_2.int_coeffs, dtype=np.int32)
  shift_arr = np.array(filt_2.b_shift, dtype=np.int32)
  filt_2_info = np.append(coeffs_arr, shift_arr)
  filt_2_info.tofile(test_dir / "coeffs_2.bin")

  out_py_int = run_py_slew(filt, sig_fl, coeffs_2)
  out_c = get_c_slew_wav(test_dir)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.mark.parametrize("filter_1", [["biquad_constant_q", 100, 8, -10], ["biquad_gain", 0], ["biquad_constant_q", 10000, 8, -10], ["biquad_gain", -10],  ["biquad_highshelf", 1000, 1, 10], ["biquad_highshelf", 1000, 1, -6], ["biquad_peaking", 1000, 0.1, 20], ["biquad_highpass", 1000, 1]])
@pytest.mark.parametrize("filter_2", [["biquad_constant_q", 100, 8, -10], ["biquad_gain", 0], ["biquad_constant_q", 10000, 8, -10], ["biquad_gain", -10],  ["biquad_highshelf", 1000, 1, 10], ["biquad_highshelf", 1000, 1, -6], ["biquad_peaking", 1000, 0.1, 20], ["biquad_highpass", 1000, 1]])
@pytest.mark.parametrize("slew_shift", [6])
def test_slew_c(in_signal, filter_1, filter_2, slew_shift):

  print(filter_1)
  print(filter_2)

  filter_type_1 = filter_1[0]
  filter_type_2 = filter_2[0]

  coeffs_hand_1 = getattr(bq, f"make_{filter_type_1}")
  coeffs_1 = coeffs_hand_1(fs, *filter_1[1:])
  coeffs_hand_2 = getattr(bq, f"make_{filter_type_2}")
  coeffs_2 = coeffs_hand_2(fs, *filter_2[1:])

  filt =bq.biquad_slew(coeffs_1, fs, 1, slew_shift=slew_shift)
  filt2 =bq.biquad_slew(coeffs_2, fs, 1, slew_shift=slew_shift)

  worker_id = os.environ.get("PYTEST_XDIST_WORKER")

  filter_name = f"{worker_id}_slew_{filter_type_1}_{filter_1[1]}_{filter_type_2}_{filter_2[1]}"
  single_slew_test(filt, filter_name, in_signal, filt2, coeffs_2)

def get_sig(len=0.05):
  sig_fl = gen.log_chirp(fs, len, 0.5)

  # sig_fl = np.ones(int(len*fs))*0.5
  sig_int = float_to_qxx(sig_fl)

  name = "slew_sig_48k"
  sig_path = bin_dir /  str(name + ".bin")
  
  xdist_safe_bin_write(sig_int, sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl, int(fs), "PCM_24")

  return sig_fl

@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()

if __name__ == "__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  test_slew_c(sig_fl, ["biquad_constant_q", 100, 8, -10], ["biquad_constant_q", 10000, 8, -10], 6)