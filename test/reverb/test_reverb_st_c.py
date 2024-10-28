# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
from .test_reverb_c import float_to_qxx, qxx_to_float
import audio_dsp.dsp.reverb_stereo as rvs
import audio_dsp.dsp.reverb_plate as rvp
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.generic as dspg
import pytest
import subprocess
from .. import test_utils as tu

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000

def get_sig_2ch(len=0.05):
  sig_l = []
  sig_l.append(gen.sin(fs, len, 997, 0.7))
  sig_l.append(gen.log_chirp(fs, len, 0.5))

  sig_fl_t = np.stack(sig_l, axis=1)
  sig_fl_t = utils.saturate_float_array(sig_fl_t, dspg.Q_SIG)

  sig_int = float_to_qxx(sig_fl_t)
  sig_path = bin_dir / "sig_2ch_48k.bin"
  tu.xdist_safe_bin_write(sig_int, sig_path)

  sf.write(gen_dir / "sig_2ch_48k.wav", sig_fl_t, int(fs), "PCM_24")
  return sig_fl_t.T

def get_c_wav(dir_name, app_name, verbose=False, sim=True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / app_name)
  stdout = subprocess.check_output(run_cmd, cwd=dir_name, shell=True)
  if verbose: print("run msg:\n", stdout)

  sig_bin = dir_name / "rv_sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)
  #deinterleave channels
  sig_int0 = sig_int[0::2]
  sig_int1 = sig_int[1::2]
  sig_int = [sig_int0, sig_int1]
  sig_int = np.stack(sig_int, axis=0)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl.T, fs, "PCM_24")
  return sig_fl

def run_py(rv, sig_fl):
  out_py = np.zeros_like(sig_fl)

  for n in range(sig_fl.shape[1]):
    out_py[:,n] = rv.process_channels_xcore(sig_fl[:,n])
  sf.write(gen_dir / "sig_py_int.wav", out_py.T, fs, "PCM_24")

  return out_py

@pytest.fixture(scope="module")
def in_signal():
    bin_dir.mkdir(exist_ok=True, parents=True)
    gen_dir.mkdir(exist_ok=True, parents=True)
    return get_sig_2ch()

@pytest.mark.parametrize("decay, damping", [[1.0, 1.0],
                                            [0.1, 0.5]
                                            ])
@pytest.mark.parametrize("wet, dry, pregain", [[-1.0, -1.0, 0.015]]) 
def test_reverb_room_st_c(in_signal, decay, damping, wet, dry, pregain):
  n_chans = 2
  max_room_size = 1.0
  room_size = 1.0
  predelay = 1
  width = 1.0

  rv = rvs.reverb_room_stereo(fs, n_chans, max_room_size, room_size, decay, damping, width, wet, dry, pregain, predelay)
  test_name = f"reverb_room_stereo_{decay}_{damping}_{wet}_{dry}_{pregain}"
  rv_info = [rv.pregain_int, rv.wet_1_int, rv.wet_2_int, rv.dry_int, rv.combs_l[0].feedback_int, rv.combs_l[0].damp1_int]
  rv_info = np.array(rv_info, dtype=np.int32)

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  rv_info.tofile(test_dir / "rv_info.bin")

  out_py_int = run_py(rv, in_signal)
  out_c = get_c_wav(test_dir, "reverb_st_test.xe")
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)

@pytest.mark.parametrize("decay, damping", [[1.0, 1.0],
                                            [0.1, 0.5]
                                            ])
@pytest.mark.parametrize("wet, dry, pregain", [[-1.0, -1.0, 0.015]]) 
def test_reverb_plate_c(in_signal, decay, damping, wet, dry, pregain):
  n_chans = 2
  predelay = 1
  #width = 1.0

  rv = rvp.reverb_plate_stereo(fs, n_chans, decay = decay, damping = damping, predelay = predelay, pregain = pregain, wet_gain_db = wet, dry_gain_db = dry)
  test_name = f"reverb_plate_{decay}_{damping}_{wet}_{dry}_{pregain}"

  # [pregain, we1, we2, dry, decay, decay_dif, damp, diffusion, bandwidth, in_dif1, in_dif2]
  rv_info = [rv.pregain_int, rv.wet_1_int, rv.wet_2_int, rv.dry_int, rv.decay_int, rv.allpasses[4].feedback_int, rv.lowpasses[1].damp1_int,
            rv.mod_allpasses[0].feedback_int, rv.lowpasses[0].damp1_int, rv.allpasses[0].feedback_int, rv.allpasses[2].feedback_int]
  rv_info = np.array(rv_info, dtype=np.int32)

  test_dir = bin_dir / test_name
  test_dir.mkdir(exist_ok = True, parents = True)

  rv_info.tofile(test_dir / "rv_info.bin")

  out_py_int = np.zeros_like(in_signal)

  # for now, runs the float implementation
  for n in range(in_signal.shape[1]):
    out_py_int[:,n] = rv.process_channels_xcore(in_signal[:,n])
  sf.write(gen_dir / "sig_py_int.wav", out_py_int.T, fs, "PCM_24")

  out_c = get_c_wav(test_dir, "reverb_plate_test.xe")
  shutil.rmtree(test_dir)

  # try switch then we have process_xcore
  np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)
  # np.testing.assert_allclose(out_c, out_py, rtol=0, atol=1e-6)
