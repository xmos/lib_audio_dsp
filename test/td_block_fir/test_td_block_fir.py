import numpy as np
from pathlib import Path
import subprocess
import sys
import shutil
import pytest
from scipy.signal import firwin
from audio_dsp.dsp.td_block_fir import generate_td_fir
from audio_dsp.dsp.ref_fir import generate_debug_fir

# TODO move build utils somewhere else
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/python')))
from build_utils import build


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
    gen_dir.mkdir(exist_ok=True, parents=True)
    # run the filter_generator on the coefs
    try:
        generate_td_fir(coefficients, "dut", gen_dir, gain_dB)
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

    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / "td_fir_test.xe") 
    
    proc = subprocess.run(run_cmd, capture_output=True, cwd = dir_name, shell = True)

    sig_int = proc.returncode

    # Clean up
    shutil.rmtree(bin_dir) 
    shutil.rmtree(gen_dir) 
    shutil.rmtree(build_dir) 

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
            
if __name__ == "__main__":
    test_random_pos_value_variable_length(2)