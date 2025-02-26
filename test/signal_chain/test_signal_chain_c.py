# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
import subprocess
import audio_dsp.dsp.signal_chain as sc
from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.signal_gen as gen
import pytest
from test.test_utils import xdist_safe_bin_write, float_to_qxx, qxx_to_float, q_convert_flt

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs = 48000


def get_sig(len=0.05):
  sig_fl = []
  sig_fl.append(gen.sin(fs, len, 997, 0.7))
  sig_fl.append(gen.sin(fs, len, 100, 0.7))
  sig_fl = np.stack(sig_fl, axis=0)
  sig_fl = q_convert_flt(sig_fl, 23, Q_SIG)

  sig_int = float_to_qxx(sig_fl)

  name = "sig_48k"
  sig_path = bin_dir /  str(name + ".bin")

  xdist_safe_bin_write(sig_int[0], sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl[0], int(fs), "PCM_24")

  name = "sig1_48k"
  sig_path = bin_dir /  str(name + ".bin")
  xdist_safe_bin_write(sig_int[1], sig_path)

  # wav file does not need to be locked as it is only used for debugging outside pytest
  wav_path = gen_dir / str(name + ".wav")
  sf.write(wav_path, sig_fl[1], int(fs), "PCM_24")

  return sig_fl


def get_c_wav(dir_name, comp_name, verbose=False, sim = True):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / f"{comp_name}_test.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  if verbose: print("run msg:\n", stdout.decode())

  sig_bin = dir_name / "sig_out.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=np.int32)

  sig_fl = qxx_to_float(sig_int)
  sf.write(gen_dir / "sig_c.wav", sig_fl, fs, "PCM_24")
  return sig_fl


def write_gain(test_dir, gain):
  all_filt_info = np.empty(0, dtype=np.int32)
  all_filt_info = np.append(all_filt_info, np.array(gain, dtype=np.int32))
  all_filt_info.tofile(test_dir / "gain.bin")


def single_channels_test(filt, test_dir, fname, sig_fl):
  out_py = np.zeros(sig_fl.shape[1])

  for n in range(sig_fl.shape[1]):
    out_py[n] = filt.process_channels_xcore(sig_fl[:, n])[0]
  
  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, fname, verbose=True)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)


@pytest.fixture(scope="module")
def in_signal():
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  return get_sig()


@pytest.mark.parametrize("gain_dB", [-10, 0, 24])
def test_gains_c(in_signal, gain_dB):
  filt = sc.fixed_gain(fs, 1, gain_dB)
  test_dir = bin_dir / f"fixed_gain_{gain_dB}"
  test_dir.mkdir(exist_ok = True, parents = True)
  write_gain(test_dir, filt.gain_int)

  out_py = np.zeros(in_signal.shape[1])
  
  for n in range(in_signal.shape[1]):
    out_py[n] = filt.process_xcore(in_signal[0][n])

  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, "fixed_gain")
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)


def test_subtractor_c(in_signal):
  filt = sc.subtractor(fs)
  test_dir = bin_dir / "subtractor"
  test_dir.mkdir(exist_ok = True, parents = True)

  single_channels_test(filt, test_dir, "subtractor", in_signal)


def test_adder_c(in_signal):
  filt = sc.adder(fs, 2)
  test_dir = bin_dir / "adder"
  test_dir.mkdir(exist_ok = True, parents = True)

  single_channels_test(filt, test_dir, "adder", in_signal)


@pytest.mark.parametrize("gain_dB", [-12, -6, 0])
def test_mixer_c(in_signal, gain_dB):
  filt = sc.mixer(fs, 2, gain_dB)
  test_dir = bin_dir / f"mixer_{gain_dB}"
  test_dir.mkdir(exist_ok = True, parents = True)
  write_gain(test_dir, filt.gain_int)

  single_channels_test(filt, test_dir, "mixer", in_signal)


@pytest.mark.parametrize("gains_dB", [[0, -6, 6], [-10, 3, 0]])
@pytest.mark.parametrize("slew", [1, 10])
@pytest.mark.parametrize("mute_test", [True, False])
def test_volume_control_c(in_signal, gains_dB, slew, mute_test):
  filt = sc.volume_control(fs, 1, gains_dB[0], slew)
  test_dir = bin_dir / f"volume_control_{gains_dB[0]}_{gains_dB[1]}_{gains_dB[2]}_{slew}_{mute_test}"
  test_dir.mkdir(exist_ok = True, parents = True)
  
  test_info = [0] * 5
  test_info[0] = mute_test
  test_info[1] = filt.slew_shift
  test_info[2] = filt.target_gain_int

  out_py = np.zeros(in_signal.shape[1])
  intervals = [0] * 4
  intervals[1] = in_signal.shape[1] // 3
  intervals[2] = intervals[1] * 2
  intervals[3] = in_signal.shape[1]
  
  for n in range(intervals[0], intervals[1]):
    out_py[n] = filt.process_xcore(in_signal[0][n])

  filt.set_gain(gains_dB[1])
  if mute_test: filt.mute()
  test_info[3] = filt.target_gain_int

  for n in range(intervals[1], intervals[2]):
    out_py[n] = filt.process_xcore(in_signal[0][n])

  filt.set_gain(gains_dB[2])
  if mute_test: filt.unmute()
  test_info[4] = filt.target_gain_int
  
  for n in range(intervals[2], intervals[3]):
    out_py[n] = filt.process_xcore(in_signal[0][n])

  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  test_info = np.array(test_info, dtype=np.int32)
  test_info.tofile(test_dir / "gain.bin")

  out_c = get_c_wav(test_dir, "volume_control")
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)


@pytest.mark.parametrize("delay_spec", [[1, 0, "samples"],
                                        [0.5, 0.5, "ms"],
                                        [0.02, 0.01, "s"],
                                        [0.5, 0, "ms"]])
def test_delay_c(in_signal, delay_spec):
  filter = sc.delay(fs, 1, *delay_spec)
  test_dir = bin_dir / f"delay_{delay_spec[0]}_{delay_spec[1]}_{delay_spec[2]}"
  test_dir.mkdir(exist_ok = True, parents = True)

  delay_info = np.empty(0, dtype=np.int32)
  delay_info = np.append(delay_info, filter._max_delay)
  delay_info = np.append(delay_info, filter._delay)
  delay_info = np.array(delay_info, dtype=np.int32)
  print(delay_info)
  delay_info.tofile(test_dir / "delay.bin")

  out_py = np.zeros((1, in_signal.shape[1]))
  for n in range(len(in_signal[0])):
    out_py[:, n] = filter.process_channels_xcore(in_signal[0, n:n+1].tolist())

  sf.write(gen_dir / "sig_py_int.wav", out_py[0], fs, "PCM_24")

  out_c = get_c_wav(test_dir, "delay")
  shutil.rmtree(test_dir)
  np.testing.assert_allclose(out_c, out_py[0], rtol=0, atol=0)


def test_switch_slew_c(in_signal):

  filt = sc.switch_slew(fs, 2)

  test_dir = bin_dir / "switch_slew"
  test_dir.mkdir(exist_ok = True, parents = True)
  fname = "switch_slew"

  out_py = np.zeros(in_signal.shape[1])

  for n in range(in_signal.shape[1]//2):
    out_py[n] = filt.process_channels_xcore(in_signal[:, n])[0]

  filt.move_switch(1)

  for n in range(in_signal.shape[1]//2, in_signal.shape[1]):
    out_py[n] = filt.process_channels_xcore(in_signal[:, n])[0]

  sf.write(gen_dir / "sig_py_int.wav", out_py, fs, "PCM_24")

  out_c = get_c_wav(test_dir, fname)
  shutil.rmtree(test_dir)

  np.testing.assert_allclose(out_c, out_py, rtol=0, atol=0)


@pytest.mark.parametrize("mix", [0, 0.1, 0.5, 0.9, 1.0])
def test_crossfader_c(in_signal, mix):
  filt = sc.crossfader(fs, 2, mix)
  test_dir = bin_dir / f"crossfader_{mix}"
  test_dir.mkdir(exist_ok = True, parents = True)
  write_gain(test_dir, filt.gains_int)

  single_channels_test(filt, test_dir, "crossfader", in_signal)


if __name__ =="__main__":
  bin_dir.mkdir(exist_ok=True, parents=True)
  gen_dir.mkdir(exist_ok=True, parents=True)
  sig_fl = get_sig()
  
  #test_gains_c(sig_fl, -6)
  #test_subtractor_c(sig_fl)
  #test_adder_c(sig_fl)
  #test_mixer_c(sig_fl, -3)
  # test_volume_control_c(sig_fl, [0, -6, 6], 7, False)
  # test_switch_slew_c(sig_fl)
  test_crossfader_c(sig_fl, 0.1)
