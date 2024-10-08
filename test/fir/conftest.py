# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
import scipy.signal as spsig
from pathlib import Path
from filelock import FileLock


def pytest_sessionstart():

    gen_dir = Path(__file__).parent / "autogen"
    gen_dir.mkdir(exist_ok=True, parents=True)

    coeffs = np.zeros(1000)
    coeffs[0] = 1
    out_dir = Path(gen_dir, "passthrough_filter.txt")
    with FileLock(str(out_dir) + ".lock"):
        if not out_dir.is_file():
            np.savetxt(out_dir, coeffs)

    coeffs = np.arange(10, 0, -1)/10
    out_dir = Path(gen_dir, "descending_coeffs.txt")
    with FileLock(str(out_dir) + ".lock"):
        if not out_dir.is_file():
            np.savetxt(out_dir, coeffs)

    coeffs = spsig.firwin2(512, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    out_dir = Path(gen_dir,"simple_low_pass.txt")
    with FileLock(str(out_dir) + ".lock"):
        if not out_dir.is_file():
            np.savetxt(out_dir, coeffs)

    coeffs = spsig.firwin2(2048, [0.0, 20/48000, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
    out_dir = Path(gen_dir, "aggressive_high_pass.txt")
    with FileLock(str(out_dir) + ".lock"):
        if not out_dir.is_file():
            np.savetxt(out_dir, coeffs)

    coeffs = spsig.firwin2(2047, [0.0, 0.5, 1.0], [0.5, 1.0, 2.0])
    out_dir = Path(gen_dir, "tilt.txt")
    with FileLock(str(out_dir) + ".lock"):
        if not out_dir.is_file():
            np.savetxt(out_dir, coeffs)

    coeffs = np.zeros(10000)
    coeffs[::8] = 1
    out_dir = Path(gen_dir, "comb.txt")
    with FileLock(str(out_dir) + ".lock"):
        if not out_dir.is_file():
            np.savetxt(out_dir, coeffs)


if __name__ == "__main__":
    pytest_sessionstart()
