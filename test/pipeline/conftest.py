# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from python import run_pipeline_xcoreai
import numpy as np
import scipy.signal as spsig
from pathlib import Path

def pytest_configure(config):
    run_pipeline_xcoreai.FORCE_ADAPTER_ID = config.getoption("--adapter-id")

def pytest_addoption(parser):
    parser.addoption(
        "--adapter-id", action="store", default=None, help="Force tests to use specific adapter"
    )

def pytest_sessionstart():

    gen_dir = Path(__file__).parent / "autogen"
    gen_dir.mkdir(exist_ok=True, parents=True)


    coeffs = np.arange(10, 0, -1)
    coeffs = coeffs/np.sum(coeffs)
    np.savetxt(Path(gen_dir, "descending_coeffs.txt"), coeffs)

    coeffs = spsig.firwin2(512, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    np.savetxt(Path(gen_dir, "simple_low_pass.txt"), coeffs)
