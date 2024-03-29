# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.generic as dspg
import audio_dsp.dsp.reverb as rv

@pytest.mark.parametrize("freq", [20, 100, 1000, 10000, 23000])
@pytest.mark.parametrize("max_room_size", [0.01, 0.1, 0.5, 1, 2, 4])
def test_reverb_overflow(freq, max_room_size):
    fs = 48000

    sig = gen.sin(fs, 5, freq, 1)
    sig = sig/np.max(np.abs(sig))
    sig = sig* (2**31 - 1)/(2**31)

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, room_size=1, decay=1.0, damping=0.0, Q_sig=30)
    print(reverb.get_buffer_lens())
    
    output_xcore = np.zeros(len(sig))
    output_flt = np.zeros(len(sig))

    for n in range(len(sig)):
        output_flt[n] = reverb.process(sig[n])

    reverb.reset_state()
    for n in range(len(sig)):
        output_xcore[n] = reverb.process_xcore(sig[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


@pytest.mark.parametrize("max_room_size", [0.01, 0.1, 0.5, 1, 2, 4])
def test_reverb_time(max_room_size):
    # measure reverb time with chirp
    fs = 48000

    sig = np.zeros(int(fs*max_room_size*40) + 1)
    sig[:1*fs] = gen.log_chirp(fs, 1, 1, 20, 20000)
    sig = sig* (2**31 - 1)/(2**31)

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, room_size=1, decay=1.0, damping=0.5, Q_sig=29)
    print(reverb.get_buffer_lens())
    
    output_xcore = np.zeros(len(sig))
    output_flt = np.zeros(len(sig))

    for n in range(len(sig)):
        output_flt[n] = reverb.process(sig[n])

    reverb.reset_state()
    for n in range(len(sig)):
        output_xcore[n] = reverb.process_xcore(sig[n])

    # check noise floor
    assert np.max(output_xcore[-1000:]) < 2**-reverb.Q_sig

    # extend by 2x
    sig = np.concatenate((sig, np.zeros_like(sig)))
    output_xcore = np.concatenate((output_xcore, np.zeros_like(output_xcore)))
    output_flt = np.concatenate((output_flt, np.zeros_like(output_flt)))

    sig_spect = np.fft.rfft(sig)
    output_xcore_spect = np.fft.rfft(output_xcore)
    output_flt_spect = np.fft.rfft(output_flt)

    # Y = HX, Y/X = H
    H_xcore_spect = output_xcore_spect/sig_spect
    h_xcore = np.fft.irfft(H_xcore_spect)
    h_xcore = h_xcore[:len(h_xcore)//2]

    H_flt_spect = output_flt_spect/sig_spect
    h_flt = np.fft.irfft(H_flt_spect)
    h_flt = h_flt[:len(h_flt)//2]


    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


def test_reverb_bypass():
    # test that a drc component is bit exact when the signal is below
    # the threshold (or above in the case of a noise gate).
    fs = 480000

    reverb = rv.reverb_room(fs, 1, dry_gain_db=0, wet_gain_db=-np.inf)

    signal = gen.log_chirp(fs, 0.5, 1)

    output_xcore = np.zeros(len(signal))
    output_flt = np.zeros(len(signal))

    for n in np.arange(len(signal)):
        output_xcore[n] = reverb.process_xcore(signal[n])
    reverb.reset_state()
    for n in np.arange(len(signal)):
        output_flt[n] = reverb.process(signal[n])


    np.testing.assert_array_equal(signal, output_flt)
    # quantization noise from multiply by dry gain
    np.testing.assert_allclose(signal, output_xcore, atol=2**-(reverb.Q_sig-1))


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("max_room_size", [0.01, 0.1, 0.5, 1, 2, 4])
def test_reverb_frames(fs, max_room_size):
    # test the process_frame functions of the drc components

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size)

    signal = gen.log_chirp(fs, 0.5, 1)
    t = np.arange(len(signal))/fs
    signal *= np.sin(t*2*np.pi*0.5)

    signal = np.tile(signal, [1, 1])
    frame_size = 1
    signal_frames = utils.frame_signal(signal, frame_size, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)

    for n in range(len(signal_frames)):
        output_int[:, n:n+frame_size] = reverb.process_frame_xcore(signal_frames[n])
    reverb.reset_state()
    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = reverb.process_frame(signal_frames[n])

    assert np.all(output_int[0, :] == output_int)
    assert np.all(output_flt[0, :] == output_flt)


if __name__ == "__main__":
    test_reverb_time(0.1)
    # test_reverb_frames(48000, 1)
