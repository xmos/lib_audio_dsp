"""
Test python vs C biquad coeff generators

Write sets of params to file (usually 3 params), then read into C.
For each parameter set, write the 5 coefficients to file from C.
Read the sets of 5 coeffs into python, compare against python implementation.
"""

import numpy as np
from pathlib import Path
import subprocess
import itertools
import time
from enum import IntEnum

import audio_dsp.dsp.biquad as bq
from audio_dsp.dsp.generic import Q_SIG

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
#   time.sleep(1)

  return sig_fl32


class bq_type(IntEnum):
    allpass = 0
    bandpass = 1
    bandstop = 2
    bypass = 3
    constq = 4
    gain = 5
    high_shelf = 6
    highpass = 7
    linkwitz = 8
    low_shelf = 9
    lowpass = 10
    mute = 11
    notch = 12 
    peaking = 13


def get_c_wav(dir_name, conv_name, verbose=True, sim = True, dtype=np.float32):

    # get the enum from the name, assuming name starts with "coeffs_..."
    bq_name = conv_name[7:]
    enum = bq_type[bq_name].value

    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / "coeffs_alltests.xe") + f" {enum}"
    stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
    if verbose: print("run msg:\n", stdout)

    sig_bin = dir_name / "out_vector.bin"
    assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
    sig_int = np.fromfile(sig_bin, dtype=dtype)

    return sig_int


def test_design_biquad_bypass():
    test_dir = bin_dir / "coeffs_bypass"
    test_dir.mkdir(exist_ok = True, parents = True)

    ratios = [[5]]
    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_bypass", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_bypass(48000)
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-16, atol=0)


def test_design_biquad_mute():
    test_dir = bin_dir / "coeffs_mute"
    test_dir.mkdir(exist_ok = True, parents = True)

    ratios = [[5]]
    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_mute", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_mute(48000)
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-16, atol=0)


def test_design_biquad_gain():
    test_dir = bin_dir / "coeffs_gain"
    test_dir.mkdir(exist_ok = True, parents = True)

    ratios = [[10], [0], [-10], [-200]]
    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_gain", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_gain(48000, ratios[n][0])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, bq.BOOST_BSHIFT)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-16, atol=0)


def test_design_biquad_lowpass():
    test_dir = bin_dir / "coeffs_lowpass"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_lowpass", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_lowpass(ratios[n][1], ratios[n][0], ratios[n][2])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-19, atol=0)


def test_design_biquad_highpass():
    test_dir = bin_dir / "coeffs_highpass"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_highpass", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_highpass(ratios[n][1], ratios[n][0], ratios[n][2])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-19, atol=0)


def bandx_param_check(ratios):
    new_ratios = []
    for ratio in ratios:
        f, fs, q = ratio
        if f < fs*1e-3:
            q = max(0.5, q)
        high_q_stability_limit = 0.85
        if q >= 5 and f/(fs/2) > high_q_stability_limit:
            f = high_q_stability_limit*fs/2
        new_ratios.append([f, fs, q])

    return new_ratios

def test_design_biquad_bandpass():
    test_dir = bin_dir / "coeffs_bandpass"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]
    ratios = bandx_param_check(ratios)

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_bandpass", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_bandpass(ratios[n][1], ratios[n][0], ratios[n][2])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-17, atol=0)


def test_design_biquad_bandstop():
    test_dir = bin_dir / "coeffs_bandstop"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]    
    ratios = list(itertools.product(f, fs, q))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]
    ratios = bandx_param_check(ratios)

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_bandstop", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_bandstop(ratios[n][1], ratios[n][0], ratios[n][2])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-17, atol=0)


def test_design_biquad_notch():
    test_dir = bin_dir / "coeffs_notch"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_notch", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_notch(ratios[n][1], ratios[n][0], ratios[n][2])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-19, atol=0)


def test_design_biquad_allpass():
    test_dir = bin_dir / "coeffs_allpass"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q))
    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]
    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_allpass", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_allpass(ratios[n][1], ratios[n][0], ratios[n][2])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-19, atol=0)


def test_design_biquad_peaking():
    test_dir = bin_dir / "coeffs_peaking"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    gain = [-12, -6, 0, 6, 18, 19]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q, gain))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_peaking", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_peaking(ratios[n][1], ratios[n][0], ratios[n][2], ratios[n][3])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, bq.BOOST_BSHIFT)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-13, atol=0)


def test_design_biquad_constq():
    test_dir = bin_dir / "coeffs_constq"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2, 5, 10]
    gain = [-40, -12, -6, 0, 6, 18, 19]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q, gain))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_constq", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_constant_q(ratios[n][1], ratios[n][0], ratios[n][2], ratios[n][3])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, bq.BOOST_BSHIFT)

    # this doesn't work if one of the coefficients is zero
    np.testing.assert_allclose(out_c[coeffs_python>0], coeffs_python[coeffs_python>0], rtol=2**-12, atol=0)


def test_design_biquad_high_shelf():
    test_dir = bin_dir / "coeffs_high_shelf"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2]
    gain = [-12, -6, 0, 6, 12, 13]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q, gain))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_high_shelf", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_highshelf(ratios[n][1], ratios[n][0], ratios[n][2], ratios[n][3])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, bq.BOOST_BSHIFT)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-12.8, atol=0)


def test_design_biquad_low_shelf():
    test_dir = bin_dir / "coeffs_low_shelf"
    test_dir.mkdir(exist_ok = True, parents = True)

    f = [20, 100, 1000, 10000, 20000]
    q = [0.1, 0.5, 1, 2]
    gain = [-12, -6, 0, 6, 12, 13]
    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    ratios = list(itertools.product(f, fs, q, gain))

    # get rid of f >= fs/2
    ratios = [ratio for ratio in ratios if ratio[0] < (ratio[1]/2)]

    flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_low_shelf", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_lowshelf(ratios[n][1], ratios[n][0], ratios[n][2], ratios[n][3])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, bq.BOOST_BSHIFT)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-12.8, atol=0)



def test_design_biquad_linkwitz():
    test_dir = bin_dir / "coeffs_linkwitz"
    test_dir.mkdir(exist_ok = True, parents = True)

    fs = [16000, 44100, 48000, 88200, 96000, 192000]
    f0 = [20, 50, 100, 200]
    fp_ratio = [0.4, 1, 4]
    q0 = [0.5, 2, 0.707]
    qp = [0.5, 2, 0.707]
    initialratios = list(itertools.product(f0, fs, q0, fp_ratio, qp))
    ratios = []
    for ratio in initialratios:
        if ratio[1] > 100000 and ratio[0] < 50 and ratio[3] < 1:
            ratio_0 = 30
        else: ratio_0 = ratio[0]
        ratios.append([ratio_0, ratio[1], ratio[2], ratio[3]*ratio_0, ratio[4]])

    ratios_32 = flt_to_bin_file(ratios, test_dir)

    out_c = get_c_wav(test_dir, "coeffs_linkwitz", dtype=np.int32)
    out_c = np.reshape(out_c, newshape=[-1, 5])
 
    coeffs_python = np.zeros((len(ratios), 5), dtype=np.int32)

    for n in range(len(ratios)):
        flt_coeffs = bq.make_biquad_linkwitz(ratios[n][1], ratios[n][0], ratios[n][2], ratios[n][3], ratios[n][4])
        _, coeffs_python[n] = bq._round_and_check(flt_coeffs, 0)

    np.testing.assert_allclose(out_c, coeffs_python, rtol=2**-22, atol=0)


if __name__ == "__main__":
    bin_dir.mkdir(exist_ok=True, parents=True)
    gen_dir.mkdir(exist_ok=True, parents=True)

    test_design_biquad_constq()
