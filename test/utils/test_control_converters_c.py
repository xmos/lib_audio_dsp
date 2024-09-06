import numpy as np
from pathlib import Path
import subprocess
import itertools
from enum import IntEnum

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.drc.drc_utils as drcu
import audio_dsp.dsp.signal_chain as sc
from audio_dsp.dsp.generic import Q_SIG, HEADROOM_DB

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs=48000

def float_to_qxx(arr_float, q = Q_SIG, dtype = np.int32):
  arr_int32 = np.clip((np.array(arr_float) * (2**q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
  return arr_int32


def qxx_to_float(arr_int, q = Q_SIG):
  arr_float = np.array(arr_int).astype(np.float64) * (2 ** (-q))
  return arr_float


def flt_to_bin_file(sig_fl, out_dir=bin_dir):
  sig_fl32 = np.array(sig_fl).astype(np.float32)
  name = "test_vector"
  sig_fl32.tofile(out_dir /  f"{name}.bin")

  return sig_fl


def get_c_wav(dir_name, conv_name, verbose=False, sim = True, dtype=np.float32):
  app = "xsim" if sim else "xrun --io"
  run_cmd = app + " " + str(bin_dir / f"{conv_name}.xe")
  stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
  if verbose: print("run msg:\n", stdout)

  sig_bin = dir_name / "out_vector.bin"
  assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
  sig_int = np.fromfile(sig_bin, dtype=dtype)

  return sig_int


def test_compressor_ratio_helper():
    test_dir = bin_dir / "compressor_ratio"
    test_dir.mkdir(exist_ok = True, parents = True)

    ratios = [0, 0.9, 1, 5, 10000, 999999999999999999999999999]
    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "compressor_ratio")

    slopes_python = np.zeros_like(ratios, dtype=np.float32)

    for n in range(len(ratios)):
        slope, slopes_python[n] = drcu.rms_compressor_slope_from_ratio(ratios[n])

    assert np.all(out_c == slopes_python)


def test_expander_ratio_helper():
    test_dir = bin_dir / "expander_ratio"
    test_dir.mkdir(exist_ok = True, parents = True)

    ratios = [0, 0.9, 1, 5, 10000, 999999999999999999999999999]
    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "expander_ratio")

    slopes_python = np.zeros_like(ratios, dtype=np.float32)

    for n in range(len(ratios)):
        slope, slopes_python[n] = drcu.peak_expander_slope_from_ratio(ratios[n])

    assert np.all(out_c == slopes_python)


def test_rms_threshold():
    test_dir = bin_dir / "rms_threshold"
    test_dir.mkdir(exist_ok = True, parents = True)

    threshold_dbs = [-2000, 0, HEADROOM_DB/2 + 1]
    flt_to_bin_file(threshold_dbs, test_dir)

    out_c = get_c_wav(test_dir, "rms_threshold", dtype=np.int32)

    thresh_python = np.zeros_like(threshold_dbs, dtype=np.int32)

    for n in range(len(threshold_dbs)):
        thresh, thresh_python[n] = drcu.calculate_rms_threshold(threshold_dbs[n], Q_SIG)

    assert np.all(out_c == thresh_python)


def test_peak_threshold():

    test_dir = bin_dir / "peak_threshold"
    test_dir.mkdir(exist_ok = True, parents = True)

    threshold_dbs = [-2000, 0, HEADROOM_DB + 1]
    flt_to_bin_file(threshold_dbs, test_dir)

    out_c = get_c_wav(test_dir, "peak_threshold", dtype=np.int32)

    thresh_python = np.zeros_like(threshold_dbs, dtype=np.int32)

    for n in range(len(threshold_dbs)):
        thresh, thresh_python[n] = drcu.calculate_threshold(threshold_dbs[n], Q_SIG, power=False)

    assert np.all(out_c == thresh_python)


def test_calc_alpha():

    test_dir = bin_dir / "calc_alpha"
    test_dir.mkdir(exist_ok = True, parents = True)

    attack_times = [-1, 0, 1/48000, 3/48000, 1, 1000, (4/48000)*(2**31)]
    flt_to_bin_file(attack_times, test_dir)

    out_c = get_c_wav(test_dir, "calc_alpha", dtype=np.int32)

    alphas_python = np.zeros_like(attack_times, dtype=np.int32)
    for n in range(len(attack_times)):
        alpha, alphas_python[n] = drcu.alpha_from_time(attack_times[n], 48000)

    # not exact due to float32 implementation differences
    np.testing.assert_allclose(out_c, alphas_python, rtol=2**-24, atol=0)


def test_db_gain():

    test_dir = bin_dir / "db_gain"
    test_dir.mkdir(exist_ok = True, parents = True)

    gain_dbs = [-2000, 0, 25]
    flt_to_bin_file(gain_dbs, test_dir)

    out_c = get_c_wav(test_dir, "db_gain", dtype=np.int32)

    gain_python = np.zeros_like(gain_dbs, dtype=np.int32)

    for n in range(len(gain_dbs)):
        gain, gain_python[n] = sc.db_to_qgain(gain_dbs[n])

    np.testing.assert_allclose(out_c, gain_python, rtol=2**-23, atol=0)


class time_units_type(IntEnum):
    samples = 0
    ms = 1
    s = 2


def test_time_samples():

    test_dir = bin_dir / "time_samples"
    test_dir.mkdir(exist_ok = True, parents = True)

    times = [10, 128, 1.7, 0.94, 2, -2]
    units = [time_units_type["samples"], time_units_type["s"], time_units_type["ms"]]

    input_params = list(itertools.product(times, units))

    flt_to_bin_file(input_params, test_dir)

    out_c = get_c_wav(test_dir, "time_samples", dtype=np.int32)

    delay_py = np.zeros(len(input_params), dtype=np.int32)
    for n in range(len(input_params)):
        delay_py[n] = utils.time_to_samples(48000, input_params[n][0], input_params[n][1].name)

    # not exact due to float32 implementation differences
    np.testing.assert_allclose(out_c, delay_py, rtol=2**-24, atol=0)


if __name__ == "__main__":
    bin_dir.mkdir(exist_ok=True, parents=True)
    gen_dir.mkdir(exist_ok=True, parents=True)

    test_db_gain()
