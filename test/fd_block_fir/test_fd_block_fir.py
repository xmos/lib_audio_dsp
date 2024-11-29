import numpy as np
from pathlib import Path
import subprocess
import os
import sys
import shutil
import pytest
from scipy.signal import firwin
from audio_dsp.dsp.fd_block_fir import generate_fd_fir
from audio_dsp.dsp.ref_fir import generate_debug_fir

# TODO move build utils somewhere else
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/python')))
from build_utils import build


build_dir_name = "build"


def build_and_run_tests(dir_name, coefficients, frame_advance = 16, td_block_length = None, frame_overlap = 0, sim = True, gain_dB = 0.0):

    local_build_dir_name = build_dir_name

    bin_dir = Path(__file__).parent / "bin"
    gen_dir = Path(__file__).parent / "autogen"
    build_dir = Path(__file__).parent / local_build_dir_name

    bin_dir.mkdir(exist_ok=True, parents=True)
    gen_dir.mkdir(exist_ok=True, parents=True)

    if frame_advance is None:
        frame_advance = max(td_block_length//2, 1)

    # run the filter_generator on the coefs
    try:
        generate_fd_fir(coefficients, "dut", gen_dir, frame_advance, frame_overlap, td_block_length, 
                      gain_dB = gain_dB, verbose = True)
        generate_debug_fir(coefficients, "dut", gen_dir, frame_advance, frame_overlap, td_block_length, 
                      gain_dB = gain_dB, verbose = True)
    except ValueError as e:
        if "Bad config" not in str(e):
            raise e
        else:
            print("caught bad config")
            print(str(e))
            print('coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
            raise e
            return
    except Exception as e:
        print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise e

    # build the project
    build(Path(dir_name), Path(build_dir), "fd_fir_test")

    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / "fd_fir_test.xe") 
    
    proc = subprocess.run(run_cmd,  cwd = dir_name, shell = True)

    sig_int = proc.returncode

    # Clean up
    shutil.rmtree(bin_dir) 
    shutil.rmtree(gen_dir) 

    if sig_int == 0:
        pass
    else:
        print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise RuntimeError(f"xsim failed: {sig_int}")

    return sig_int

dir_name = Path(__file__).parent

def test_trivial():
    build_and_run_tests(dir_name, np.random.uniform(-0.125, 0.125, 34))

# @pytest.mark.parametrize("td_block_length", [16])
@pytest.mark.parametrize("filter_length_mul", [1, 2, 3])
@pytest.mark.parametrize("filter_length_mod", [-2, 1, 3])
@pytest.mark.parametrize("frame_overlap", [0, 3])
@pytest.mark.parametrize("frame_advance_mod", [-2, 0, 1])
def test_constant_value_variable_length(td_block_length, filter_length_mul, filter_length_mod, frame_overlap, frame_advance_mod):
    filter_length = (td_block_length*filter_length_mul)//2 + filter_length_mod
    frame_advance = td_block_length//2 + frame_advance_mod
    build_and_run_tests(dir_name, 
                        np.ones(filter_length)/filter_length, 
                        td_block_length = None, 
                        frame_overlap = frame_overlap,
                        frame_advance = frame_advance)

@pytest.mark.parametrize("length", range(15, 19, 2))
def test_random_value_variable_length(length):
    build_and_run_tests(dir_name, 0.125*np.random.uniform(-1, 1, length))

@pytest.mark.parametrize("length", range(1, 18, 2))
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

@pytest.mark.skip("Slow test")
@pytest.mark.parametrize("length", [1024, 4096])
def test_long_lengths(length):
    build_and_run_tests(dir_name, np.random.uniform(-1, 1, length))

@pytest.mark.parametrize("length", [16, 17, 18, 32, 33, 34, 127, 128, 129])
def test_real_filter(length):
    build_and_run_tests(dir_name, firwin(length, 0.5))

if __name__ == "__main__":
    # test_constant_value_variable_length(16, 2, -2, 2, 0)
    # test_long_lengths(1024)
    test_trivial()