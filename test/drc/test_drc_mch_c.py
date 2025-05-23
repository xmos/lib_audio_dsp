# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import audio_dsp.dsp.drc as drc
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.generic as dspg
import pytest

from test.test_utils import xdist_safe_bin_write, float_to_qxx, q_convert_flt
from test_drc_c import get_c_wav

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000

def get_sig_2ch(len=0.05):
  sig_l = []
  sig_l.append(gen.sin(fs, len, 997, 0.7))
  sig_l.append(gen.square(fs, len, 50, 0.5) + 0.5)
  sig_fl_t = np.stack(sig_l, axis=1)
  sig_fl_t = q_convert_flt(sig_fl_t, 23, dspg.Q_SIG)

  sig_int = float_to_qxx(sig_fl_t)

  name = "sig_2ch_48k"
  sig_path = bin_dir /  str(name + ".bin")

  xdist_safe_bin_write(sig_int, sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl_t, int(fs), "PCM_24")
  
  return sig_fl_t.T

@pytest.fixture(scope="module")
def in_signal_2ch():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig_2ch()


def get_sig_4ch(len=0.05):
  sig_l = []
  sig_l.append(gen.sin(fs, len, 997, 0.7))
  sig_l.append(gen.sin(fs, len, 435, 0.7))
  sig_l.append(gen.square(fs, len, 50, 0.5) + 0.5)
  sig_l.append(gen.square(fs, len, 55, 0.5) + 0.5)
  sig_fl_t = np.stack(sig_l, axis=1)
  sig_fl_t = q_convert_flt(sig_fl_t, 23, dspg.Q_SIG)

  sig_int = float_to_qxx(sig_fl_t)

  name = "sig_4ch_48k"
  sig_path = bin_dir /  str(name + ".bin")

  xdist_safe_bin_write(sig_int, sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl_t, int(fs), "PCM_24")
  
  return sig_fl_t.T

@pytest.fixture(scope="module")
def in_signal_4ch():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig_4ch()

@pytest.mark.parametrize("comp_name", ["compressor_rms_sidechain_mono"])
@pytest.mark.parametrize("at", [0.001])
@pytest.mark.parametrize("rt", [0.01])
@pytest.mark.parametrize("threshold", [-12, 0])
@pytest.mark.parametrize("ratio", [1, 6])
def test_sidechain_c(in_signal_2ch, comp_name, at, rt, threshold, ratio):
  component_handle = getattr(drc, comp_name)
  comp = component_handle(fs, ratio, threshold, at, rt)
  test_name = f"{comp_name}_{ratio}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  # numpy doesn't like to have an array with different types
  # so create separate arrays, cast to bytes, append, write
  comp_info = [comp.threshold_int, comp.attack_alpha_int, comp.release_alpha_int]
  comp_info = np.array(comp_info, dtype=np.int32)
  comp_info1 = np.array(comp.slope_f32, dtype=np.float32)
  comp_info = comp_info.tobytes()
  comp_info1 = comp_info1.tobytes()
  comp_info = np.append(comp_info, comp_info1)
  comp_info.tofile(test_dir / "info.bin")

  out_py = np.zeros(in_signal_2ch.shape[1])
  for n in range(in_signal_2ch.shape[1]):
    out_py[n], _, _ = comp.process_xcore(in_signal_2ch[0][n], in_signal_2ch[1][n])
  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, comp_name)
  shutil.rmtree(test_dir)

  # when ratio is 1, the result should be bit-exact as we don't have to use powf
  if ratio == 1 or threshold == 0:
    np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)
  else:
    # tolerace is the 24b float32 mantissa
    tol = 2**(np.ceil(np.log2(np.max(out_c))) - 24)
    np.testing.assert_allclose(out_c, out_py, rtol=0, atol=tol)


@pytest.mark.parametrize("comp_name", ["compressor_rms_sidechain_stereo"])
@pytest.mark.parametrize("at", [0.001])
@pytest.mark.parametrize("rt", [0.01])
@pytest.mark.parametrize("threshold", [-12, 0])
@pytest.mark.parametrize("ratio", [1, 6])
def test_sidechain_stereo_c(in_signal_4ch, comp_name, at, rt, threshold, ratio):
  component_handle = getattr(drc, comp_name)
  comp = component_handle(fs, ratio, threshold, at, rt)
  test_name = f"{comp_name}_{ratio}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  # numpy doesn't like to have an array with different types
  # so create separate arrays, cast to bytes, append, write
  comp_info = [comp.threshold_int, comp.attack_alpha_int, comp.release_alpha_int]
  comp_info = np.array(comp_info, dtype=np.int32)
  comp_info1 = np.array(comp.slope_f32, dtype=np.float32)
  comp_info = comp_info.tobytes()
  comp_info1 = comp_info1.tobytes()
  comp_info = np.append(comp_info, comp_info1)
  comp_info.tofile(test_dir / "info.bin")

  out_py = np.zeros((in_signal_4ch.shape[1], 2))
  for n in range(in_signal_4ch.shape[1]):
    out_py[n], _, _ = comp.process_channels_xcore(in_signal_4ch[0:2, n], in_signal_4ch[2:, n])
  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, comp_name)
  out_c_deinter = np.array([out_c[0::2], out_c[1::2]]).T
  shutil.rmtree(test_dir)

  # when ratio is 1, the result should be bit-exact as we don't have to use powf
  if ratio == 1 or threshold == 0:
    np.testing.assert_allclose(out_c_deinter, out_py, rtol=0, atol=0)
  else:
    # tolerace is the 24b float32 mantissa
    tol = 2**(np.ceil(np.log2(np.max(out_c))) - 24)
    np.testing.assert_allclose(out_c_deinter, out_py, rtol=0, atol=tol)

if __name__ == "__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig_4ch()
  
  test_sidechain_stereo_c(sig_fl, "compressor_rms_sidechain_stereo", 0.001, 0.01, -6, 4)
