# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest
from functools import partial
import warnings

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.generic as dspg
import audio_dsp.dsp.reverb as rv


@pytest.mark.parametrize("max_room_size", [0.1, 0.5, 1, 2, 4])
@pytest.mark.parametrize("signal, freq", [["sine", 20],
                                          ["sine", 1000],
                                          ["sine", 1000],
                                          ["sine", 10000],
                                          ["sine", 23000],
                                          ["noise", None]])
def test_reverb_overflow(signal, freq, max_room_size):
    # check no overflow errors occur
    fs = 48000
    q_format = 31

    if signal == "sine":
        sig = gen.sin(fs, 5, freq, 1)
    elif signal == "chirp":
        sig = gen.log_chirp(fs, 5, 1, 20, 20000)
    elif signal == "noise":
        sig = gen.white_noise(fs, 5, 1)

    sig = sig/np.max(np.abs(sig))
    sig = sig* (2**31 - 1)/(2**31)

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, room_size=1, decay=1.0, damping=0.0, Q_sig=q_format)
    #print(reverb.get_buffer_lens())
    
    output_xcore = np.zeros(len(sig))
    output_flt = np.zeros(len(sig))

    for n in range(len(sig)):
        output_flt[n] = reverb.process(sig[n])

    reverb.reset_state()
    for n in range(len(sig)):
        output_xcore[n] = reverb.process_xcore(sig[n])


def calc_reverb_time(in_sig, reverb_output):
    # extend by 2x
    sig = np.concatenate((in_sig, np.zeros_like(in_sig)))
    output_xcore = np.concatenate((reverb_output, np.zeros_like(reverb_output)))

    sig_spect = np.fft.rfft(sig)
    output_xcore_spect = np.fft.rfft(output_xcore)

    # Y = HX, Y/X = H
    H_xcore_spect = output_xcore_spect/sig_spect
    h_xcore = np.fft.irfft(H_xcore_spect)
    h_xcore = h_xcore[:len(h_xcore)//2]

    return h_xcore


@pytest.mark.parametrize("max_room_size", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("decay", [0, 0.5, 1])
@pytest.mark.parametrize("damping", [0, 0.5])
@pytest.mark.parametrize("q_format, pregain", [[27, 0.015],
                                               [31, 0.001]])
def test_reverb_time(max_room_size, decay, damping, q_format, pregain):
    # measure reverb time with chirp
    fs = 48000

    sig = np.zeros(int(fs*max_room_size*6) + fs)
    sig[:1*fs] = gen.log_chirp(fs, 1, 1, 20, 20000)
    sig = sig* (2**q_format - 1)/(2**q_format)

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, room_size=1, decay=decay, damping=damping, Q_sig=q_format, pregain=pregain)
    
    output_xcore = np.zeros(len(sig))
    output_flt = np.zeros(len(sig))

    for n in range(len(sig)):
        output_flt[n] = reverb.process(sig[n])

    reverb.reset_state()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', utils.SaturationWarning)
        for n in range(len(sig)):
            output_xcore[n] = reverb.process_xcore(sig[n])

    # if we triggered a saturation warning, can't guarantee arrays are the same
    sat_warn_flag = all([wi.category is utils.SaturationWarning for wi in w])

    # # in this case, pregain should be adjusted
    # if sat_warn_flag: assert False

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = np.logical_and(utils.db(output_flt) > -50, utils.db(output_flt) < (6*(31-q_format)))
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


@pytest.mark.parametrize("max_room_size", [0.5])
@pytest.mark.parametrize("decay", [0.5])
@pytest.mark.parametrize("damping", [0.5])
def test_reverb_noise_floor(max_room_size, decay, damping):
    # check the reverb decays to 0 (no limit cycle noise)
    fs = 48000
    q_format = 27

    sig = np.zeros(int(fs*max_room_size*40) + fs)
    sig[:1*fs] = gen.log_chirp(fs, 1, 1, 20, 20000)
    sig = sig* (2**q_format - 1)/(2**q_format)

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, room_size=1, decay=decay, damping=damping, Q_sig=q_format)
    #print(reverb.get_buffer_lens())
    
    output_xcore = np.zeros(len(sig))
    output_flt = np.zeros(len(sig))

    for n in range(len(sig)):
        output_flt[n] = reverb.process(sig[n])

    reverb.reset_state()
    with warnings.catch_warnings(record=True) as w:
        for n in range(len(sig)):
            output_xcore[n] = reverb.process_xcore(sig[n])

    # check noise floor
    assert np.max(np.abs(output_xcore[-1000:])) < 2**-(reverb.Q_sig + 1)

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = np.logical_and(utils.db(output_flt) > -50, utils.db(output_flt) < (6*(31-q_format)))
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
@pytest.mark.parametrize("max_room_size", [0.01, 0.1, 0.5, 2, 4])
@pytest.mark.parametrize("q_format", [27, 31])
def test_reverb_frames(fs, max_room_size, q_format):
    # test the process_frame functions of the drc components

    # if q_format > 27:
    #     pytest.xfail("This test is not meant to pass with a q more then 27")

    reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, Q_sig=q_format)

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

def test_reverb_properties_decay():
    """Basic tests to check for consistency when setting the properties."""
    r = partial(rv.reverb_room, 48000, 1)

    val = 0.1
    a = r(decay=val)
    b = r()
    b.decay = val
    c = r()
    c.set_decay(val)

    should_be_val = np.array([i.decay for i in (a, b, c)])
    np.testing.assert_allclose(should_be_val, val)

def test_reverb_properties_pregain():
    """Basic tests to check for consistency when setting the properties."""
    r = partial(rv.reverb_room, 48000, 1)

    val = 0.1
    a = r(pregain=val)
    b = r()
    b.pregain = val
    c = r()
    c.set_pre_gain(val)

    should_be_val = np.array([i.pregain for i in (a, b, c)])
    np.testing.assert_allclose(should_be_val, val)

def test_reverb_properties_wet_db():
    """Basic tests to check for consistency when setting the properties."""
    r = partial(rv.reverb_room, 48000, 1)

    val = -6
    a = r(wet_gain_db=val)
    b = r()
    b.wet_db = val
    c = r()
    c.set_wet_gain(val)

    should_be_val = np.array([i.wet_db for i in (a, b, c)])
    np.testing.assert_allclose(should_be_val, val)

def test_reverb_properties_dry_db():
    """Basic tests to check for consistency when setting the properties."""
    r = partial(rv.reverb_room, 48000, 1)

    val = -6
    a = r(dry_gain_db=val)
    b = r()
    b.dry_db = val
    c = r()
    c.set_dry_gain(val)

    should_be_val = np.array([i.dry_db for i in (a, b, c)])
    np.testing.assert_allclose(should_be_val, val)

def test_reverb_properties_damping():
    """Basic tests to check for consistency when setting the properties."""
    r = partial(rv.reverb_room, 48000, 1)

    val = 0.5
    a = r(damping=val)
    b = r()
    b.damping = val
    c = r()
    c.set_damping(val)

    should_be_val = np.array([i.damping for i in (a, b, c)])
    np.testing.assert_allclose(should_be_val, val)

def test_reverb_properties_room_size():
    """Basic tests to check for consistency when setting the properties."""
    r = partial(rv.reverb_room, 48000, 1)

    val = 0.5
    a = r(room_size=val)
    b = r()
    b.room_size = val
    c = r()
    c.set_room_size(val)

    should_be_val = np.array([i.room_size for i in (a, b, c)])
    np.testing.assert_allclose(should_be_val, val)

if __name__ == "__main__":
    test_reverb_overflow("sine", 20, 0.01)
    # test_reverb_time(0.01, 1)
    # test_reverb_frames(48000, 1)
