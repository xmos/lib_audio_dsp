from scipy import signal
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import audio_dsp.dsp.biquad as bq
from audio_dsp.dsp.generic import Q_SIG
import pytest

build_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000


def float_to_qxx(arr_float, q = Q_SIG, dtype = np.int32):
  arr_int32 = np.clip((np.array(arr_float) * (2**q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
  return arr_int32


def qxx_to_float(arr_int, q = Q_SIG):
  arr_float = np.array(arr_int).astype(np.float64) * (2 ** (-q))
  return arr_float


def get_sig(len = 0.25):
  time = np.arange(0, len, 1 / fs)
  sig_fl = 0.5 * signal.chirp(time, 20, time[-1], 0.8 * fs / 2, "log", phi = -90)
  sig_int = float_to_qxx(sig_fl)
  # sig_fl should be as quantised as sig_int for bit-exactnes
  sig_fl = qxx_to_float(sig_int)
  name = "sig_48k"
  sig_int.tofile(build_dir /  str(name + ".bin"))
  sf.write(gen_dir / str(name + ".wav"), sig_fl, int(fs), "PCM_24")

  return sig_fl


def get_c_wav(run = True, bin_name = "sig_out.bin", sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(build_dir / "biquad_test.xe")
  if run:
    stdout = subprocess.check_output(run_cmd, cwd = build_dir, shell = True)
    #print("run msg:\n", stdout)

  sig_bin = build_dir / bin_name
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)

  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def run_py(filt, sig_fl):
  out_int = np.zeros(sig_fl.size)
  out_fl = np.zeros(sig_fl.size)
  
  for n in range(sig_fl.size):
    out_int[n] = filt.process_vpu(sig_fl[n])

  sf.write(gen_dir / "sig_py_int.wav", out_int, fs, "PCM_24")
  filt.reset_state()

  for n in range(sig_fl.size):
    out_fl[n] = filt.process(sig_fl[n])

  sf.write(gen_dir / "sig_py_flt.wav", out_fl, fs, "PCM_24")

  return out_fl, out_int


def write_c_coeffs(filt, fname):
  coeffs_arr = np.array(filt.int_coeffs, dtype=np.int32)
  shift_arr = np.array(filt.b_shift, dtype=np.int32)
  filt_info = np.append(coeffs_arr, shift_arr)
  filt_info.tofile(build_dir / fname)


@pytest.mark.parametrize("filter_type", ["biquad_lowpass",
                                         "biquad_highpass",
                                         "biquad_notch",
                                         "biquad_allpass"])
@pytest.mark.parametrize("f", [20, 200, 2000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.707, 5, 10])
def test_xpass_filters_c(filter_type, f, q):
  build_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)

  f = np.min([f, fs / 2 * 0.95])
  filter_handle = getattr(bq, filter_type)
  filt = filter_handle(fs, f, q)
  write_c_coeffs(filt, "coeffs.bin")

  sig_fl = get_sig()
  out_py_fl, out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav()

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.mark.parametrize("filter_type", ["biquad_peaking",
                                         "biquad_constant_q",
                                         "biquad_lowshelf",
                                         "biquad_highshelf",])
@pytest.mark.parametrize("f", [20, 200, 2000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.707, 5, 10])
@pytest.mark.parametrize("gain", [-12, -5, 0, 5, 12])
def test_high_gain_c(filter_type, f, q, gain):
  build_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)

  f = np.min([f, fs / 2 * 0.95])
  filter_handle = getattr(bq, filter_type)
  filt = filter_handle(fs, f, q, gain)
  write_c_coeffs(filt, "coeffs.bin")

  sig_fl = get_sig()
  out_py_fl, out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav()

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.mark.parametrize("filter_type", ["biquad_bandpass",
                                         "biquad_bandstop",])
@pytest.mark.parametrize("f", [20, 200, 2000, 20000])
@pytest.mark.parametrize("q", [0.1, 0.707, 5, 10])
def test_bandx_filters_c(filter_type, f, q):
  build_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)

  filter_handle = getattr(bq, filter_type)
  f = np.min([f, fs / 2 * 0.95])
  high_q_stability_limit = 0.85
  if q >= 5 and f / (fs / 2) > high_q_stability_limit:
    f = high_q_stability_limit * fs / 2
  filt = filter_handle(fs, f, q)
  write_c_coeffs(filt, "coeffs.bin")

  sig_fl = get_sig()
  out_py_fl, out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav()

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


@pytest.mark.parametrize("f0", [20, 100, 500])
@pytest.mark.parametrize("fp", [20, 100, 500])
@pytest.mark.parametrize("q0", [0.5, 1, 2])
@pytest.mark.parametrize("qp", [0.5, 1, 2])
def test_linkwitz_filters_c(f0, fp, q0, qp):
  build_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)

  filt = bq.biquad_linkwitz(fs, f0, q0, fp, qp)
  write_c_coeffs(filt, "coeffs.bin")

  sig_fl = get_sig()
  out_py_fl, out_py_int = run_py(filt, sig_fl)
  out_c = get_c_wav()

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)


if __name__ =="__main__":
  test_linkwitz_filters_c(100, 500, 0.5, 1)
