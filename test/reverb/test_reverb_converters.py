# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest
import numpy as np
from audio_dsp.dsp.reverb_base import Q_VERB
from audio_dsp.dsp.reverb import reverb_room
from audio_dsp.dsp.reverb_stereo import reverb_room_stereo
from audio_dsp.dsp.reverb_plate import reverb_plate_stereo
from subprocess import run
from pathlib import Path
import os

CWD = Path(__file__).parent
TOL_KWARGS = dict(rtol=2**-16, atol=0)
LESS_THAN_1 = ((2**Q_VERB) - 1) / (2**Q_VERB)


def q_verb(x):
    return int(x * 2**Q_VERB)


def db2lin(db):
    return 10 ** (db / 20)


def new_reverb(**kwargs):
    return reverb_room(48000, 1, **kwargs)


def get_c(config, val):
    bin_dir = CWD / "bin" / config
    out_dir = CWD / "bin" / f"{config}_{val}"
    os.makedirs(out_dir, exist_ok=True)
    sig_fl32 = np.array(val).astype(np.float32)
    name = "test_vector"
    sig_fl32.tofile(out_dir / f"{name}.bin")

    xe = bin_dir / f"reverb_converters_{config}.xe"
    run(["xsim", str(xe)], check=True, cwd=out_dir)
    #print(out_dir)
    return np.fromfile(out_dir / "out_vector.bin", dtype=np.int32)


def db2int(db):
    return q_verb(db2lin(db))


def get_output(config, input, sattr, gattr):
    c_val = get_c(config, input)[0]
    r = new_reverb()
    setattr(r, sattr, input)
    p_val = getattr(r, gattr)
    return c_val, p_val


@pytest.mark.parametrize(
    "sattr,gattr",
    [
        ["wet_db", "wet_int"],
        ["dry_db", "dry_int"],
    ],
)
@pytest.mark.parametrize(
    "input,expected",
    [
        [-6, db2int(-6)],
        [0, q_verb(1)],
        [1, q_verb(1)],
        [6, q_verb(1)],
    ],
)
def test_reverb_db2int(sattr, gattr, input, expected):
    np.testing.assert_allclose(
        get_output("DB2INT", input, sattr, gattr), expected, **TOL_KWARGS
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        [0, q_verb(0.7)],
        [-1, q_verb(0.7)],
        [1, q_verb(0.98)],
        [1.1, q_verb(0.98)],
    ],
)
def test_reverb_decay2feedback(input, expected):
    np.testing.assert_allclose(
        get_output("DECAY2FEEDBACK", input, "decay", "feedback_int"),
        expected,
        **TOL_KWARGS,
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        [-0.5, 0],
        [0, 0],
        [0.5, q_verb(0.5)],
        [LESS_THAN_1, q_verb(LESS_THAN_1)],
        [1, q_verb(LESS_THAN_1)],
        [2, q_verb(LESS_THAN_1)],
    ],
)
def test_reverb_float2int(input, expected):
    np.testing.assert_allclose(
        get_output("FLOAT2INT", input, "pregain", "pregain_int"), expected, **TOL_KWARGS
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        [-0.5, 1],
        [0, 1],
        [0.5, q_verb(0.5)],
        [LESS_THAN_1, q_verb(LESS_THAN_1)],
        [1, q_verb(LESS_THAN_1)],
        [2, q_verb(LESS_THAN_1)],
    ],
)
def test_reverb_damping(input, expected):
    np.testing.assert_allclose(
        get_output("CALCULATE_DAMPING", input, "damping", "damping_int"),
        expected,
        **TOL_KWARGS,
    )

@pytest.mark.parametrize(
    "input", [-1, 0, 0.07, 0.23, 0.5, 0.64, 0.92, 1, 2]
)
def test_reverb_wet_dry_mix_conv(input):
    c_vals = get_c("WET_DRY_MIX", input)
    r = new_reverb()
    r.set_wet_dry_mix(input)
    p_vals = np.array([r.dry_int, r.wet_int], dtype=np.int32)
    # -23 cause of the float to int conversion
    np.testing.assert_allclose(c_vals, p_vals, rtol=2**-23, atol=0)

@pytest.mark.parametrize(
    "input", [-1, 0, 0.07, 0.23, 0.5, 0.64, 0.92, 1, 2]
)
def test_reverb_st_wet_dry_mix_conv(input):
    c_vals = get_c("WET_DRY_MIX_ST", input)
    r = reverb_room_stereo(48000, 2)
    r.set_wet_dry_mix(input)
    p_vals = np.array([r.dry_int, r.wet_1_int, r.wet_2_int], dtype=np.int32)
    # -23 cause of the float to int conversion
    np.testing.assert_allclose(c_vals, p_vals, rtol=2**-23, atol=0)

@pytest.mark.parametrize(
    "input", [-1, 100, 1000, 2000, 4000, 8000, 16000, 32000]
)
def test_reverb_cutoff(input):
    c_vals = get_c("CUTOFF", input)
    r = reverb_plate_stereo(48000, 2)
    r.bandwidth = input
    p_vals = r.lowpasses[0].coeff_b0_int
    # -23 cause of the float to int conversion
    np.testing.assert_allclose(c_vals, p_vals, rtol=2**-13, atol=0)


if __name__ == "__main__":
    test_reverb_cutoff(8000)