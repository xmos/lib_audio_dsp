import numpy as np
from pathlib import Path
import subprocess
import sys
import shutil
import pytest
from scipy.signal import firwin
# I dont know how to do this properly
sys.path.append('../../python/audio_dsp/dsp/')
from audio_dsp.dsp.td_block_fir import process_array

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
        process_array(coefficients, "dut", gen_dir, gain_dB, debug = True, silent = True)
    except ValueError as e:
        # print('Success (Expected Fail)')
        print('coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise e
    except Exception as e:
        # print('Fail', repr(error))
        print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise e
    
    # build the project
    subprocess.check_output("cmake -B " + build_dir_name, cwd = dir_name, shell = True, stderr = subprocess.DEVNULL)
    subprocess.check_output("xmake -C " + build_dir_name, cwd = dir_name, shell = True)
    
    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / "td_fir_test.xe") 
    
    proc = subprocess.run(run_cmd, capture_output=True, cwd = dir_name, shell = True)

    sig_int = proc.returncode

    # Clean up
    shutil.rmtree(bin_dir) 
    shutil.rmtree(gen_dir) 
    shutil.rmtree(build_dir) 

    if sig_int == 0:
        # print("Success")
        pass
    else:
        # print("Fail")
        print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
    return sig_int

dir_name = Path(__file__).parent

def test_trivial():
    build_and_run_tests(dir_name, np.ones(1))

@pytest.mark.parametrize("length", range(2, 17))
def test_constant_value_variable_length(length):
    build_and_run_tests(dir_name, np.ones(length))

@pytest.mark.parametrize("length", range(2, 17))
def test_random_value_variable_length(length):
    build_and_run_tests(dir_name, np.random.uniform(-1, 1, length))

@pytest.mark.parametrize("length", range(2, 17))
def test_extreme_value_variable_length(length):
    c = np.random.randint(0, 2, length)*2 - 1
    build_and_run_tests(dir_name, c)

@pytest.mark.parametrize("length", range(2, 17))
def test_all_negative_variable_length(length):
    c = -np.ones(length)
    build_and_run_tests(dir_name, c)

@pytest.mark.parametrize("length", range(2, 17))
def test_random_pos_value_variable_length(length):
    build_and_run_tests(dir_name, np.abs(np.random.uniform(-1, 1, length)))

@pytest.mark.parametrize("length", range(2, 17))
def test_random_neg_value_variable_length(length):
    build_and_run_tests(dir_name, np.abs(np.random.uniform(-1, 1, length)))

@pytest.mark.parametrize("length", [128, 1024, 4096])
@pytest.mark.parametrize("length_mod", [-1, 0, 1])
def test_long_lengths(length, length_mod):
    build_and_run_tests(dir_name, np.random.uniform(-1, 1, length+length_mod))

@pytest.mark.parametrize("length", [127, 128, 129])
def test_real_filter(length):
    build_and_run_tests(dir_name, firwin(length, 0.5))
            
