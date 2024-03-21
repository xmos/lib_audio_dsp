# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
from test_drc_c import float_to_qxx, get_c_wav
import audio_dsp.dsp.drc as drc
import audio_dsp.dsp.signal_gen as gen
import pytest


bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000

def get_sig_2ch(dir_name, len=0.05):
  sig_l = []
  sig_l.append(gen.sin(fs, len, 997, 0.7))
  sig_l.append(gen.square(fs, len, 50, 0.5) + 0.5)
  sig_fl = np.stack(sig_l, axis=0)
  sig_fl_sf = np.stack(sig_l, axis=1)
  sig_int = float_to_qxx(sig_fl)

  name = "ch0"
  sig_int[0].tofile(dir_name / f"{name}.bin")
  name = "ch1"
  sig_int[1].tofile(dir_name / f"{name}.bin")

  sf.write(gen_dir / "sig_48k.wav", sig_fl_sf, int(fs), "PCM_24")
  return sig_fl

@pytest.mark.parametrize("comp_name", ["compressor_rms_sidechain_mono"])
@pytest.mark.parametrize("at", [0.001])
@pytest.mark.parametrize("rt", [0.01])
@pytest.mark.parametrize("threshold", [-12, 0])
@pytest.mark.parametrize("ratio", [1, 6])
def test_sidechain_c(comp_name, at, rt, threshold, ratio):
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  component_handle = getattr(drc, comp_name)
  comp = component_handle(fs, ratio, threshold, at, rt)
  test_name = f"{comp_name}_{ratio}_{threshold}_{at}_{rt}"

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)
  sig_fl = get_sig_2ch(test_dir)

  # numpy doesn't like to have an array with different types
  # so create separate arrays, cast to bytes, append, write
  comp_info = [comp.threshold_int, comp.attack_alpha_int, comp.release_alpha_int]
  comp_info = np.array(comp_info, dtype=np.int32)
  comp_info1 = np.array(comp.slope_f32, dtype=np.float32)
  comp_info = comp_info.tobytes()
  comp_info1 = comp_info1.tobytes()
  comp_info = np.append(comp_info, comp_info1)
  comp_info.tofile(test_dir / "comp_info.bin")

  out_py = np.zeros(sig_fl.shape[1])
  for n in range(sig_fl.shape[1]):
    out_py[n], _, _ = comp.process_xcore(sig_fl[0][n], sig_fl[1][n])
  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, comp_name)
  shutil.rmtree(test_dir)

  # when ratio is 1, the result should be bit-exact as we don't have to use powf
  if ratio == 1 or threshold == 0:
    np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)
  else:
    np.testing.assert_allclose(out_c, out_py, rtol=0, atol=1.5e-8)

if __name__ == "__main__":
  test_sidechain_c("compressor_rms_sidechain_mono", 0.001, 0.01, -6, 4)
