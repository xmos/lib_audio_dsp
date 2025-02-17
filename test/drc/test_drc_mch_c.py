# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
from .test_drc_c import float_to_qxx, get_c_wav
import audio_dsp.dsp.drc as drc
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.generic as dspg
import pytest
from .. import test_utils as tu

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000

def get_sig_2ch(len=0.05):
  sig_l = []
  sig_l.append(gen.sin(fs, len, 997, 0.7))
  sig_l.append(gen.square(fs, len, 50, 0.5) + 0.5)
  sig_fl_t = np.stack(sig_l, axis=1)
  sig_fl_t = utils.saturate_float_array(sig_fl_t, dspg.Q_SIG)

  sig_int = float_to_qxx(sig_fl_t)

  name = "sig_2ch_48k"
  sig_path = bin_dir /  str(name + ".bin")

  tu.xdist_safe_bin_write(sig_int, sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl_t, int(fs), "PCM_24")
  
  return sig_fl_t.T

@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig_2ch()

@pytest.mark.parametrize("comp_name", ["compressor_rms_sidechain_mono"])
@pytest.mark.parametrize("at", [0.001])
@pytest.mark.parametrize("rt", [0.01])
@pytest.mark.parametrize("threshold", [-12, 0])
@pytest.mark.parametrize("ratio", [1, 6])
def test_sidechain_c(in_signal, comp_name, at, rt, threshold, ratio):
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

  out_py = np.zeros(in_signal.shape[1])
  for n in range(in_signal.shape[1]):
    out_py[n], _, _ = comp.process_xcore(in_signal[0][n], in_signal[1][n])
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

if __name__ == "__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig_2ch()
  
  test_sidechain_c(sig_fl, "compressor_rms_sidechain_mono", 0.001, 0.01, -6, 4)
