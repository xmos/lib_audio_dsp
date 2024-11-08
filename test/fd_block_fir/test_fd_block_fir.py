import numpy as np
from pathlib import Path
import subprocess
import sys
import shutil
from scipy.signal import firwin
from audio_dsp.dsp.fd_block_fir import process_array

build_dir_name = "build"


def build_and_run_tests(dir_name, coefficients, frame_advance = None, td_block_length = 32, frame_overlap = 0, sim = True, gain_dB = 0.0):

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
        process_array(coefficients, "dut", gen_dir, frame_advance, frame_overlap, td_block_length, 
                      gain_dB = gain_dB, debug = True, warn = False, error = False, verbose = False)
    except ValueError as e:
        # print('Success (Expected Fail)')
        print('coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise e
    except Exception as e:
        # print('Fail', repr(error))
        print('FAIL coef count', len(coefficients), 'frame_advance', frame_advance, 'td_block_length', td_block_length, 'frame_overlap', frame_overlap)
        raise e

    # build the project
    subprocess.check_output("cmake -B " + local_build_dir_name, cwd = dir_name, shell = True, stderr = subprocess.DEVNULL)
    subprocess.check_output("xmake -C " + local_build_dir_name, cwd = dir_name, shell = True)
    
    app = "xsim" if sim else "xrun --io"
    run_cmd = app + " --args " + str(bin_dir / "fd_fir_test.xe") 
    
    proc = subprocess.run(run_cmd,  cwd = dir_name, shell = True)

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
    build_and_run_tests(dir_name, np.random.uniform(-0.125, 0.125, 34))

def test_constant_value_variable_length():
    for td_block_length in [16, 32]:
        for filter_length_mul in [1, 2, 3]:
            for filter_length_mod in [-2, -1, 0, 1, 2, 3]:
                filter_length = (td_block_length*filter_length_mul)//2 + filter_length_mod
                for frame_overlap in range(0, 4):
                    for frame_advance_mod in [-2, -1, 0, 1]:
                        frame_advance = td_block_length//2 + frame_advance_mod
                        build_and_run_tests(dir_name, 
                                            np.ones(filter_length)/filter_length, 
                                            td_block_length = td_block_length, 
                                            frame_overlap = frame_overlap,
                                            frame_advance = frame_advance)

def test_random_value_variable_length():
    for length in range(15, 19):
        build_and_run_tests(dir_name, 0.125*np.random.uniform(-1, 1, length))

def test_extreme_value_variable_length():
    for length in range(1, 18):
        c = np.random.randint(0, 2, length)*2 - 1
        build_and_run_tests(dir_name, c)

def test_all_negative_variable_length():
    for length in range(2, 17):
        c = -np.ones(length)
        build_and_run_tests(dir_name, c)

def test_random_pos_value_variable_length():
    for length in range(2, 17):
        build_and_run_tests(dir_name, np.abs(np.random.uniform(-1, 1, length)))

def test_random_neg_value_variable_length():
    for length in range(2, 17):
        build_and_run_tests(dir_name, np.abs(np.random.uniform(-1, 1, length)))

def test_long_lengths():
    for length in [1024, 4096]:
        build_and_run_tests(dir_name, np.random.uniform(-1, 1, length))

def test_real_filter():
    for length in [16, 17, 18, 32, 33, 34, 127, 128, 129]:
        build_and_run_tests(dir_name, firwin(length, 0.5))
            
if __name__ == "__main__":

    print("test_trivial")
    test_trivial()
    print("test_constant_value_variable_length")
    test_constant_value_variable_length()
    print("test_real_filter")
    test_real_filter()
    print("test_random_neg_value_variable_length")
    test_random_neg_value_variable_length()
    print("test_random_pos_value_variable_length")
    test_random_pos_value_variable_length()
    print("test_all_negative_variable_length")
    test_all_negative_variable_length()
    print("test_extreme_value_variable_length")
    test_extreme_value_variable_length()
    print("test_random_value_variable_length")
    test_random_value_variable_length()