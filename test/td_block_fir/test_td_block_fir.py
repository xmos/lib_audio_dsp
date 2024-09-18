"""
Test python vs C biquad coeff generators

Write sets of params to file (usually 3 params), then read into C.
For each parameter set, write the 5 coefficients to file from C.
Read the sets of 5 coeffs into python, compare against python implementation.
"""

import numpy as np
from pathlib import Path
import subprocess

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"

fs=48000



from "../../python/audio_dsp/dsp/td_block_fir.py" import process_array

def build_and_run_tests(dir_name, coefficients, verbose=True, sim = True):

    output_path = "src"
    gain_dB = 0.0

    # run the filter_generator on the coefs
    process_array(coefficients, "dut", output_path, gain_dB, debug = True)

    # build the project
    stdout = subprocess.check_output("xmake", cwd = dir_name, shell = True)

    # run xsim
    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / "test_td_block_fir.xe") 
    stdout = subprocess.check_output(run_cmd, cwd = dir_name, shell = True)
    if verbose: print("run msg:\n", stdout)

    sig_bin = dir_name / "out_vector.bin"
    assert sig_bin.is_file(), f"Could not find output bin {sig_bin}"
    sig_int = np.fromfile(sig_bin, dtype=int)

    return sig_int

def test_trivial():
    coefs = np.ones(1)
    build_and_run_tests(dir_name, coefs)
    np.testing.assert_true(True)

if __name__ == "__main__":
    bin_dir.mkdir(exist_ok=True, parents=True)
    gen_dir.mkdir(exist_ok=True, parents=True)
    test_trivial()
