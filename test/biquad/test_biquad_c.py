# Copyright 2024-2025 XMOS LIMITED.
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
  run_cmd = app + " " + str(bin_dir / "biquad_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  #print("run msg:\n", stdout)

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt: bq.biquad, sig_fl):
  out_int = np.zeros(sig_fl.size)
  
  for n in range(sig_fl.size):
    out_int[n] = filt.process_xcore(sig_fl[n])

  sf.write(gen_dir / "sig_py_int.wav", out_int, fs, "PCM_24")

  return out_int


def single_test(filt, tname, sig_fl):
  test_dir = bin_dir / tname
  test_dir.mkdir(exist_ok = True, parents = True)
  coeffs_arr = np.array(filt.int_coeffs, dtype=np.int32)
  shift_arr = np.array(filt.b_shift, dtype=np.int32)
  filt_info = np.append(coeffs_arr, shift_arr)
  filt_info.tofile(test_dir / "coeffs.bin")

  out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav(test_dir)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()


@pytest.mark.parametrize("filter_type", ["biquad_lowpass",
                                         "biquad_highpass",
                                         "biquad_notch",
                                         "biquad_allpass"])
@pytest.mark.parametrize("f", [20, 20000])
@pytest.mark.parametrize("q", [0.1, 10])
def test_xpass_filters_c(in_signal, filter_type, f, q):

  f = np.min([f, fs / 2 * 0.95])
  filter_handle = getattr(bq, "make_%s" % filter_type)

  filt = bq.biquad(filter_handle(fs, f, q), fs, 1)
  filter_name = f"{filter_type}_{f}_{q}"
  single_test(filt, filter_name, in_signal)


@pytest.mark.parametrize("filter_type", ["biquad_peaking",
                                         "biquad_constant_q",
                                         "biquad_lowshelf",
                                         "biquad_highshelf",])
@pytest.mark.parametrize("f", [20, 20000])
@pytest.mark.parametrize("q", [0.1, 10])
@pytest.mark.parametrize("gain", [-12, 12])
def test_high_gain_c(in_signal, filter_type, f, q, gain):

  f = np.min([f, fs / 2 * 0.95])
  filter_handle = getattr(bq, "make_%s" % filter_type)

  filt = bq.biquad(filter_handle(fs, f, q, gain), fs, 1)
  filter_name = f"{filter_type}_{f}_{q}_{gain}"
  single_test(filt, filter_name, in_signal)


@pytest.mark.parametrize("filter_type", ["biquad_bandpass",
                                         "biquad_bandstop",])
@pytest.mark.parametrize("f", [20, 20000])
@pytest.mark.parametrize("q", [0.1, 10])
def test_bandx_filters_c(in_signal, filter_type, f, q):

  f = np.min([f, fs / 2 * 0.95])
  high_q_stability_limit = 0.85
  if q >= 5 and f / (fs / 2) > high_q_stability_limit:
    f = high_q_stability_limit * fs / 2
  filter_handle = getattr(bq, "make_%s" % filter_type)

  filt = bq.biquad(filter_handle(fs, f, q), fs, 1)
  filter_name = f"{filter_type}_{f}_{q}"
  single_test(filt, filter_name, in_signal)


@pytest.mark.parametrize("f0", [20, 100, 500])
@pytest.mark.parametrize("fp_ratio", [0.4, 4])
@pytest.mark.parametrize("q0, qp", [(0.5, 2), (2, 0.5), (0.707, 0.707)])
def test_linkwitz_filters_c(in_signal, f0, fp_ratio, q0, qp):

  fp = f0*fp_ratio
  filt = bq.biquad(bq.make_biquad_linkwitz(fs, f0, q0, f0*fp_ratio, qp), fs, 1)
  filter_name = f"biquad_linkwitz_{f0}_{fp_ratio}_{q0}_{qp}"
  single_test(filt, filter_name, in_signal)

@pytest.mark.parametrize("gain", [-10, 0, 10])
def test_gain_filters_c(in_signal, gain):
  
  filt = bq.biquad(bq.make_biquad_gain(fs, gain), fs, 1)
  filter_name = f"biquad_gain_{gain}"
  single_test(filt, filter_name, in_signal)


if __name__ =="__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  #test_xpass_filters_c(sig_fl, "biquad_notch", 200, 0.7)
  #test_high_gain_c(sig_fl, "biquad_lowshelf", 2000, 0.1, 5)
  test_bandx_filters_c(sig_fl, "biquad_bandpass", 200, 10)
  #test_linkwitz_filters_c(sig_fl, 100, 4, 0.5, 2)
  # test_gain_filters_c(sig_fl, -10)
