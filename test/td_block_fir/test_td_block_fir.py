import numpy as np
from pathlib import Path
import subprocess
import sys
import shutil
import pytest
from scipy.signal import firwin
from audio_dsp.dsp.td_block_fir import generate_td_fir

import uuid
from filelock import FileLock

# TODO move build utils somewhere else
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/python')))
from build_utils import build
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../fd_block_fir')))
from ref_fir import generate_debug_fir

build_dir_name = "build"

bin_dir = Path(__file__).parent / "bin"
gen_dir = Path(__file__).parent / "autogen"
build_dir = Path(__file__).parent / build_dir_name


def build_and_run_tests(dir_name, coefficients, frame_advance = 8, td_block_length = 8, frame_overlap = 0, sim = True, gain_dB = 0.0):

    local_build_dir_name = build_dir_name

    bin_dir = Path(__file__).parent / "bin"
    gen_dir = Path(__file__).parent / "autogen"
    build_dir = Path(__file__).parent / local_build_dir_name

    bin_dir.mkdir(exist_ok=True, parents=True)
    # the builds share files, so can't be built in parallel, but we can run xsim in parallel after
    with FileLock("build_blocker.lock"):
        gen_dir.mkdir(exist_ok=True, parents=True)

        # run the filter_generator on the coefs
        try:
            generate_td_fir(coefficients, "dut", gen_dir, gain_dB=gain_dB)
            generate_debug_fir(coefficients, "dut", gen_dir, gain_dB = gain_dB, verbose = False)
        except ValueError as e:
            # print('Success (Expected Fail)')
            print('coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
            raise e
        except Exception as e:
            # print('Fail', repr(error))
            print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
            raise e
        
        # build the project
        build(Path(dir_name), Path(build_dir), "td_fir_test")
    
        unique_xe = str(bin_dir / f"{uuid.uuid4().hex[:10]}_td_fir_test.xe")
        os.rename(str(bin_dir / "td_fir_test.xe"), unique_xe)

        # Clean up
        shutil.rmtree(gen_dir) 

    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / unique_xe) 
    
    proc = subprocess.run(run_cmd, capture_output=True, cwd = dir_name, shell = True)

    sig_int = proc.returncode

    if sig_int == 0:
        pass
    else:
        print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise RuntimeError(f"xsim failed: {sig_int}")
    return sig_int

dir_name = Path(__file__).parent

def test_trivial():
    build_and_run_tests(dir_name, np.ones(1))

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_constant_value_variable_length(length):
    build_and_run_tests(dir_name, np.ones(length))

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_random_value_variable_length(length):
    build_and_run_tests(dir_name, np.random.uniform(-1, 1, length))

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_extreme_value_variable_length(length):
    c = np.random.randint(0, 2, length)*2 - 1
    build_and_run_tests(dir_name, c)

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_all_negative_variable_length(length):
    c = -np.ones(length)
    build_and_run_tests(dir_name, c)

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_random_pos_value_variable_length(length):
    build_and_run_tests(dir_name, np.abs(np.random.uniform(-1, 1, length)))

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_random_neg_value_variable_length(length):
    build_and_run_tests(dir_name, np.abs(np.random.uniform(-1, 1, length)))

@pytest.mark.parametrize("length", [127, 128, 129])
def test_real_filter(length):
    build_and_run_tests(dir_name, firwin(length, 0.5))

@pytest.mark.parametrize("length", range(2, 17, 2))
def test_main(length):
    coeffs = np.abs(np.random.uniform(-1, 1, length))
    coeff_name = f"tmp_coeffs_{length}.npy"
    np.save(coeff_name, coeffs)

    subprocess.check_output(f"python -m audio_dsp.dsp.td_block_fir {coeff_name} --output 'autogen'", shell=True)


if __name__ == "__main__":
    # test_random_pos_value_variable_length(2)
    test_main(10)