# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest
import numpy as np

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.types as types

# seed RNG to avoid collection failures
np.random.seed(0)

@pytest.mark.parametrize('x', (2*np.random.rand(100)-1)*2**64)
def test_float_s32_float(x):
    x_s32 = types.float_s32(float(x))
    x2 = float(x_s32)

    tol = 2**x_s32.exp
    assert np.isclose(x, x2, rtol=2**-31, atol=tol)


@pytest.mark.parametrize('x, Q', np.stack(((2*np.random.rand(100)-1),
                                           np.random.randint(0, 30, size=100)),
                                          axis=-1))
def test_float_s32_float_Q(x, Q):
    x_s32 = types.float_s32(float(x), Q_sig=Q)
    x2 = float(x_s32)

    tol = 2**-Q
    assert np.isclose(x, x2, rtol=2**-31, atol=tol)


@pytest.mark.parametrize('x', (2*np.random.rand(100)-1)*2**64)
def test_float_s32_abs(x):
    x_s32 = types.float_s32(float(x))
    x2 = float(abs(x_s32))

    tol = 2**x_s32.exp
    assert np.isclose(abs(x), x2, rtol=2**-31, atol=tol)


@pytest.mark.parametrize("x, y", (2*np.random.rand(100, 2)-1)*2**30)
def test_float_s32_mult(x, y):
    x_s32 = types.float_s32(float(x))
    y_s32 = types.float_s32(float(y))

    xy_s32 = x_s32*y_s32
    xy = float(xy_s32)

    tol = 2**xy_s32.exp
    assert np.isclose(x*y, xy, rtol=2**-29, atol=tol)



@pytest.mark.parametrize("x, y", (2*np.random.rand(100, 2)-1)*2**30)
def test_float_s32_div(x, y):
    x_s32 = types.float_s32(float(x))
    y_s32 = types.float_s32(float(y))

    xy_s32 = x_s32/y_s32
    xy = float(xy_s32)

    tol = 2**xy_s32.exp
    assert np.isclose(x/y, xy, rtol=2**-29, atol=tol)


@pytest.mark.parametrize("x, y", (2*np.random.rand(100, 2)-1)*2**30)
def test_float_s32_add(x, y):
    x_s32 = types.float_s32(float(x))
    y_s32 = types.float_s32(float(y))

    xy_s32 = x_s32 + y_s32
    xy = float(xy_s32)

    tol = 2**(max(x_s32.exp, y_s32.exp) + 2)
    assert np.isclose(x + y, xy, rtol=2**-29, atol=tol)


@pytest.mark.parametrize("x, y", (2*np.random.rand(100, 2)-1)*2**30)
def test_float_s32_subt(x, y):
    x_s32 = types.float_s32(float(x))
    y_s32 = types.float_s32(float(y))

    xy_s32 = x_s32-y_s32
    xy = float(xy_s32)

    tol = 2**(max(x_s32.exp, y_s32.exp) + 1)
    assert np.isclose(x-y, xy, rtol=2**-29, atol=tol)


@pytest.mark.parametrize("x, y", (2*np.random.rand(100, 2)-1)*2**30)
def test_float_s32_gt(x, y):
    x_s32 = types.float_s32(float(x))
    y_s32 = types.float_s32(float(y))

    xy_s32 = x_s32 > y_s32
    xy = x > y

    assert xy == xy_s32


@pytest.mark.parametrize("x, y", (2*np.random.rand(100, 2)-1)*2**30)
def test_float_s32_lt(x, y):
    x_s32 = types.float_s32(float(x))
    y_s32 = types.float_s32(float(y))

    xy_s32 = x_s32 < y_s32
    xy = x < y

    assert xy == xy_s32


if __name__ == "__main__":
    for n in range(100):
        # test_float_s32_float((2*np.random.rand(1)-1)*2**32)
        test_float_s32_add((2*np.random.rand(1)-1)*2**27,
                           (2*np.random.rand(1)-1)*2**27)
