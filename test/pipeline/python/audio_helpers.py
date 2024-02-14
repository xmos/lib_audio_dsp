
"""
Helper functions for creating and validating audio files
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

from audio_dsp.dsp import generic

def generate_test_signal(wav_file_name, type="sine", fs=48000, duration=10, num_channels=2, amplitude=0.8, sig_dtype=np.int32):
    if type == "sine":
        f = 1000
        sig = np.empty((int(fs*duration), num_channels))
        sample_space = np.linspace(0, duration, int(fs*duration))

        for i in range(num_channels):
            f_sig = f * (i+1) # Generate harmonics of 1 KHz
            sig[:,i] = (amplitude * np.sin(2 * np.pi * f_sig * sample_space)).T

        if (sig_dtype == np.int32) or (sig_dtype == np.int16):
            sig = np.array(sig * np.iinfo(sig_dtype).max, dtype=sig_dtype)
        scipy.io.wavfile.write(wav_file_name, fs, sig)
    else:
        assert False, f"ERROR: Generating {type} signal not supported"

def read_wav(path):
    rate, data = scipy.io.wavfile.read(path)
    return rate, data

def write_wav(path, fs, data):
    return scipy.io.wavfile.write(path, fs, data)

def read_and_truncate(path, f_bits=generic.Q_SIG):
    """Read wav and truncate the least significant fractional bits"""
    rate, data = read_wav(path)

    if data.dtype != np.int32:
        raise TypeError(f"wav data type {data.dtype} not supported")

    mask = ~int(2**(31 - f_bits) - 1)
    print("mask ", mask)
    return rate, data & mask

def correlate_and_diff(output_file, input_file, out_ch_start_end, in_ch_start_end, skip_seconds_start, skip_seconds_end, tol, corr_plot_file=None, verbose=False):
    rate_out, data_out = scipy.io.wavfile.read(output_file)
    rate_in, data_in = scipy.io.wavfile.read(input_file)

    if data_out.ndim == 1:
        data_out = data_out.reshape(len(data_out), 1)

    if data_in.ndim == 1:
        data_in = data_in.reshape(len(data_in), 1)

    if rate_out != rate_in:
        assert False, f"input and output file rates are not equal. input rate {rate_in}, output rate {rate_out}"


    if data_in.dtype != np.int32:
        if data_in.dtype == np.int16:
            data_in = np.array(data_in, dtype=np.int32)
            data_in = data_in * (2**16)
        else:
            assert False, "Unsupported data_in.dtype {data_in.dtype}"

    if data_out.dtype != np.int32:
        if data_out.dtype == np.int16:
            data_out = np.array(data_out, dtype=np.int32)
        else:
            assert False, "Unsupported data_out.dtype {data_out.dtype}"

    assert out_ch_start_end[1]-out_ch_start_end[0] == in_ch_start_end[1]-in_ch_start_end[0], "input and output files have different channel nos."


    skip_samples_start = int(rate_out * skip_seconds_start)
    skip_samples_end = int(rate_out * skip_seconds_end)
    data_in = data_in[:,in_ch_start_end[0]:in_ch_start_end[1]+1]
    data_out = data_out[:,out_ch_start_end[0]:out_ch_start_end[1]+1]

    small_len = min(len(data_in), len(data_out), 64000)
    data_in_small = data_in[skip_samples_start : small_len+skip_samples_start, :].astype(np.float64)
    data_out_small = data_out[skip_samples_start : small_len+skip_samples_start, :].astype(np.float64)

    corr = scipy.signal.correlate(data_in_small[:, 0], data_out_small[:, 0], "full")
    delay = (corr.shape[0] // 2) - np.argmax(corr)
    print(f"delay = {delay}")

    if corr_plot_file != None:
        plt.plot(corr)
        plt.savefig(corr_plot_file)
        plt.clf()
    delay_orig = delay

    data_size = min(data_in.shape[0], data_out.shape[0])
    data_size -= skip_samples_end

    print(f"compare {data_size - skip_samples_start} samples")

    num_channels = out_ch_start_end[1]-out_ch_start_end[0]+1
    all_close = True
    max_diff = []
    for ch in range(num_channels):
        print(f"comparing ch {ch}")
        close = np.isclose(
                    data_in[skip_samples_start : data_size - delay, ch],
                    data_out[skip_samples_start + delay : data_size, ch],
                    atol=tol,
                )
        print(f"ch {ch}, close = {np.all(close)}")

        if verbose:
            int_max_idxs = np.argwhere(close[:] == False)
            print("shape = ", int_max_idxs.shape)
            print(int_max_idxs)
            if np.all(close) == False:
                if int_max_idxs[0] != 0:
                    count = 0
                    for i in int_max_idxs:
                        if count < 100: # Print first 100 values that were not close
                            print(i, data_in[skip_samples_start+i, ch], data_out[skip_samples_start + delay + i, ch])
                            count += 1

        diff = np.abs((data_in[skip_samples_start : data_size - delay, ch]) - (data_out[skip_samples_start + delay : data_size, ch]))
        max_diff.append(np.amax(diff))
        print(f"max diff value is {max_diff[-1]}")
        all_close = all_close & np.all(close)

    print(f"all_close: {np.all(all_close)}")
    return all_close, max(max_diff), delay_orig



