# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from audio_dsp.dsp.generic import Q_SIG
import audio_dsp.dsp.reverb as reverb
import audio_dsp.dsp.signal_gen as gen
import numpy as np
from pathlib import Path
import pytest
import shutil
import soundfile as sf
import subprocess
from filelock import FileLock

BIN_DIR = Path(__file__).parent / "bin"
GEN_DIR = Path(__file__).parent / "autogen"
FS = 48000


def float_to_qxx(arr_float, q=Q_SIG, dtype=np.int32):
    arr_int32 = np.clip(
        (np.array(arr_float) * (2**q)), np.iinfo(dtype).min, np.iinfo(dtype).max
    ).astype(dtype)
    return arr_int32


def qxx_to_float(arr_int, q=Q_SIG):
    arr_float = np.array(arr_int).astype(np.float64) * (2 ** (-q))
    return arr_float


def get_sig(len=0.05):
    sig_fl = gen.log_chirp(FS, len, 0.5)
    sig_int = float_to_qxx(sig_fl)

    name = "rv_sig_48k"
    sig_path = BIN_DIR /  str(name + ".bin")

    with FileLock(str(sig_path) + ".lock"):
        if not sig_path.is_file():
            sig_int.tofile(sig_path)

    wav_path = GEN_DIR / str(name + ".wav")
    with FileLock(str(wav_path) + ".lock"):
        if not wav_path.is_file():
            sf.write(wav_path, sig_fl, int(FS), "PCM_32")
    return sig_fl


def get_c_wav(dir_name, app_name, verbose=False, sim=True):
    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " " + str(BIN_DIR / app_name)
    stdout = subprocess.check_output(run_cmd, cwd=dir_name, shell=True)
    if verbose: print("run msg:\n", stdout)

    sig_bin = dir_name / "rv_sig_out.bin"
    assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
    sig_int = np.fromfile(sig_bin, dtype=np.int32)

    sig_fl = qxx_to_float(sig_int)
    sf.write(GEN_DIR / "sig_c.wav", sig_fl, FS, "PCM_32")
    return sig_fl


def run_py(uut: reverb.reverb_room, sig_fl, use_float_sig=True):
    out_int = np.zeros(sig_fl.size)
    out_fl = np.zeros(sig_fl.size)
    sig_int = float_to_qxx(sig_fl)

    if use_float_sig:
        for n in range(sig_fl.size):
            out_int[n] = uut.process_xcore(sig_fl[n])
    else:
        for n in range(sig_fl.size):
            out_int[n] = uut.process_xcore(sig_int[n])
        out_int = qxx_to_float(out_int)

    sf.write(GEN_DIR / "sig_py_int.wav", out_int, FS, "PCM_32")
    uut.reset_state()

    for n in range(sig_fl.size):
        out_fl[n] = uut.process(sig_fl[n])

    sf.write(GEN_DIR / "sig_py_flt.wav", out_fl, FS, "PCM_32")

    return out_int


@pytest.fixture(scope="module")
def in_signal():
    BIN_DIR.mkdir(exist_ok=True, parents=True)
    GEN_DIR.mkdir(exist_ok=True, parents=True)
    return get_sig()

@pytest.mark.parametrize("decay, damping", [[1.0, 1.0], [0.1, 0.5]])
@pytest.mark.parametrize("wet, dry, pregain", [[-1.0, -1.0, 0.015]]) 
def test_reverb_room(in_signal, decay, damping, wet, dry, pregain):
    n_chans = 1
    fs = FS
    max_room_size = 1.0
    room_size = 1.0
    predelay = 10

    uut = reverb.reverb_room(fs, n_chans, max_room_size, room_size, decay, damping, wet, dry, pregain, predelay)
    test_name = f"reverb_room_{decay}_{damping}_{wet}_{dry}_{pregain}"

    test_dir = BIN_DIR / test_name
    test_dir.mkdir(exist_ok = True, parents = True)

    rv_info = [uut.pregain_int, uut.wet_int, uut.dry_int, uut.combs[0].feedback_int, uut.combs[0].damp1_int]
    rv_info = np.array(rv_info, dtype=np.int32)
    rv_info.tofile(test_dir / "rv_info.bin")

    out_py_int = run_py(uut, in_signal)
    out_c = get_c_wav(test_dir, "reverb_test.xe")
    shutil.rmtree(test_dir)

    np.testing.assert_allclose(out_c, out_py_int, rtol=0, atol=0)
