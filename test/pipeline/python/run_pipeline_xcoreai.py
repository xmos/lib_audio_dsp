# Copyright 2022-2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import xscope_fileio
import argparse
import shutil
import subprocess
import scipy.io.wavfile
import pathlib
from filelock import FileLock
import time

FORCE_ADAPTER_ID = None

class XtagError(RuntimeError):
    pass

def xtag_getter():
    timeout = time.time() + 60*10   # 10 minutes from now
    while True:
        time.sleep(0.1)
        if time.time() > timeout:
            raise RuntimeError("No Xtag available before timeout")
        with FileLock("xtag.lock"):
            try:
                adapter_id = get_adapter_id()
                print("Got XTAG %s" % adapter_id)
                return adapter_id
            except XtagError:
                continue

def get_adapter_id():
    # check the --adapter-id option
    if FORCE_ADAPTER_ID is not None:
        return FORCE_ADAPTER_ID

    try:
        xrun_out = subprocess.check_output(['xrun', '-l'], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('Error: %s' % e.output)
        assert False

    xrun_out = xrun_out.split('\n')
    # Check that the first 4 lines of xrun_out match the expected lines
    expected_header = ["", "Available XMOS Devices", "----------------------", ""]
    if len(xrun_out) < len(expected_header):
        raise RuntimeError(
            f"Error: xrun output:\n{xrun_out}\n"
            f"does not contain expected header:\n{expected_header}"
        )

    header_match = True
    for i, expected_line in enumerate(expected_header):
        if xrun_out[i] != expected_line:
            header_match = False

    if not header_match:
        raise RuntimeError(
            f"Error: xrun output header:\n{xrun_out[:4]}\n"
            f"does not match expected header:\n{expected_header}"
        )

    try:
        if "No Available Devices Found" in xrun_out[4]:
            raise XtagError(f"Error: No available devices found\n")
            return
    except IndexError:
        raise RuntimeError(f"Error: xrun output is too short:\n{xrun_out}\n")

    for line in xrun_out[6:]:
        if line.strip():
            adapterID = line[26:34].strip()
            status = line[34:].strip()
        else:
            continue
    print("adapter_id = ",adapterID)
    return adapterID


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("xe", nargs='?',
                        help=".xe file to run")
    parser.add_argument('-i', '--input', type=pathlib.Path, required=True, help="input wav file")
    parser.add_argument('-o', '--output', type=pathlib.Path, required=True, help="output wav file")
    parser.add_argument('-n', '--num-out-channels', type=str, help="Number of channels in the output file. If unspecified, set to be same as the input number of channels",
                        default=None)

    args = parser.parse_args()
    return args

def run(xe, input_file, output_file, num_out_channels, pipeline_stages=1, return_stdout=False):
    """
    Run an xe on an xcore with fileio enabled.

    Parameters
    ----------
    xe : str
        file to run
    input_file : str
        wav file to play
    output_file : str
        output wav file
    num_out_channels : int
        number of channels the pipeline will output.
    pipeline_stages : int
        Number of stages in the pipeline, the frame delay from the input to the output. This
        is used to synchronise inputs and outputs. Set to 1 to get unmodified output from the
        pipeline.
    return_stdout : bool
        if true the process output will be returned
    """
    # Create the cmd line string
    args = f"-i {input_file} -o {output_file} -n {num_out_channels} -t {pipeline_stages - 1}"
    with open("args.txt", "w") as fp:
        fp.write(args)

    adapter_id = xtag_getter()
    print("Running on adapter_id ",adapter_id)

    if return_stdout == False:
        xscope_fileio.run_on_target(adapter_id, xe)
    else:
        with open("stdout.txt", "w+") as ff:
            xscope_fileio.run_on_target(adapter_id, xe, stdout=ff)
            ff.seek(0)
            stdout = ff.readlines()
        return stdout

if __name__ == "__main__":
    args = parse_arguments()
    assert args.xe is not None, "Specify vaild .xe file"
    print(f"args.input = {args.input}")

    if(args.num_out_channels == None):
        rate, data = scipy.io.wavfile.read(args.input)
        if data.ndim == 1:
            args.num_out_channels = 1
        else:
            args.num_out_channels = data.shape[1]

    print(f"num_out_channels = {args.num_out_channels}")
    run(args.xe, args.input, args.output, args.num_out_channels)

