# Copyright 2023 XMOS LIMITED.
# This Software is subject to the terms of the XCORE VocalFusion Licence.

from pathlib import Path
from platform import system
import os
import shutil
import subprocess
from random import randint, random

def get_dummy_files():
    control_protocol = "i2c"
    dl_prefix = ""
    dl_suffix = ""
    bin_suffix = ""

    system_name = system()
    if system_name == "Linux":
        dl_prefix = "lib"
        dl_suffix = ".so"
    elif system_name == "Darwin":
        dl_prefix = "lib"
        dl_suffix = ".dylib"
    elif system_name == "Windows":
        dl_suffix = ".dll"
        bin_suffix = ".exe"
    else:
        assert 0, "Unsupported operating system"

    host_bin = "dsp_host" + bin_suffix
    cmd_map_so = dl_prefix + "command_map"
    device_so = dl_prefix + "device_"
    local_build_folder = Path(__file__).parents[1] / "build"
    build_dir = local_build_folder if local_build_folder.is_dir() else Path(__file__).parents[1]
    test_dir = build_dir / "test"
    host_bin_path = build_dir / host_bin
    host_bin_copy = test_dir / host_bin
    cmd_map_dummy_path = test_dir / (cmd_map_so + "_dummy" + dl_suffix)
    cmd_map_path = test_dir / (cmd_map_so + dl_suffix)
    device_dummy_path = test_dir / (device_so + "dummy" + dl_suffix)
    device_path = test_dir / (device_so + control_protocol + dl_suffix)

    assert host_bin_path.is_file() or host_bin_copy.is_file(), f"host app binary not found here {host_bin}"
    if (not host_bin_copy.is_file()) or (host_bin_path.is_file() and host_bin_copy.is_file()):
        shutil.copy2(host_bin_path,  host_bin_copy)

    assert cmd_map_dummy_path.is_file() or cmd_map_path.is_file(), f"not found {cmd_map_dummy_path}"
    if (not cmd_map_path.is_file()) or (cmd_map_dummy_path.is_file() and cmd_map_path.is_file()):
        if cmd_map_path.is_file():
            os.remove(cmd_map_path)
        os.rename(cmd_map_dummy_path, cmd_map_path)

    assert device_dummy_path.is_file() or device_path.is_file(), f"not found {device_dummy_path}"
    if (not device_path.is_file()) or (device_dummy_path.is_file() and device_path.is_file()):
        if device_path.is_file():
            os.remove(device_path)
        os.rename(device_dummy_path, device_path)
    return test_dir, host_bin_copy, control_protocol, cmd_map_so + dl_suffix

def run_cmd(command, cwd, verbose = False, expect_success = True):
    result = subprocess.run(command, capture_output=True, cwd=cwd, shell=True)

    if verbose or result.returncode:
        print('\n')
        print("cmd: ", result.args)
        print("returned: ", result.returncode)
        print("stdout: ", result.stdout)
        print("stderr: ", result.stderr)

    if expect_success:
        assert not result.returncode
        return result.stdout
    else:
        assert result.returncode
        return result.stderr

def execute_command(host_bin, control_protocol, cwd, cmd_name, cmd_map_path = None, cmd_vals = None, expect_success = True):

    command = str(host_bin) + " -u " + control_protocol + " " + cmd_name
    if cmd_map_path:
        print(f"cmd_map_path in execute_command() is {cmd_map_path}")
        command = str(host_bin) + " -u " + control_protocol + " -cmp " + cmd_map_path + " " + cmd_name
    if cmd_vals != None:
        cmd_write = command + " " + ' '.join(str(val) for val in cmd_vals)
        run_cmd(cmd_write, cwd, True, expect_success)

    stdout = run_cmd(command, cwd, True)
    words = str(stdout, 'utf-8').strip().split(' ')

    # This will check that the right command is returned
    if cmd_name[0] != "-" :
        assert words[0] == cmd_name

        # Second word should be the value. Return as string so caller must cast to right type
        if len(words) == 2: # cmd and 1 value
            return words[1] # To avoid changing all the other tests that call execute_command and expect a single value and not an array
        else:
            return words[1:]
    else:
        return words

def gen_rand_array(type, min, max, size=20):
    vals = []
    vals = [0 for i in range (size)]
    if type == "float":
        vals = [random() * (max - min) + min for i in range(size)]
    elif type == "int":
        vals = [randint(min, max) for i in range(size)]
    else:
        print('Unknown type: ', type)
    return vals
