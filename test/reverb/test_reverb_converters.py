import pytest
import numpy as np
from audio_dsp.dsp.reverb import reverb_room, Q_VERB
from subprocess import run
from pathlib import Path

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
    out_dir = CWD / "bin" / config
    sig_fl32 = np.array(val).astype(np.float32)
    name = "test_vector"
    sig_fl32.tofile(out_dir / f"{name}.bin")

    xe = out_dir / f"reverb_converters_{config}.xe"
    run(["xsim", str(xe)], check=True, cwd=out_dir)
    print(out_dir)
    return np.fromfile(out_dir / "out_vector.bin", dtype=np.int32)[0]


def db2int(db):
    return q_verb(db2lin(db))


def get_output(config, input, sattr, gattr):
    c_val = get_c(config, input)
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