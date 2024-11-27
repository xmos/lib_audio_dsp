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
import audio_dsp.dsp.reverb_stereo as rvs
import audio_dsp.dsp.reverb_plate as rvp


@pytest.mark.parametrize("signal, freq", [["sine", 20],
                                          ["sine", 1000],
                                          ["sine", 10000],
                                          ["sine", 23000],
                                          ["noise", None]])
@pytest.mark.parametrize("algo, param", [["mono_room", 0.1],
                                         ["mono_room", 1],
                                         ["mono_room", 4],
                                         ["stereo_room", 0.1],
                                         ["stereo_room", 1],
                                         ["stereo_room", 4],
                                         ["stereo_plate", 0.95],]
                                         )
def test_reverb_overflow(signal, freq, algo, param):
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

    if algo == "stereo_room":
        sig = np.tile(sig, [2, 1])
        reverb = rvs.reverb_room_stereo(fs, 2, max_room_size=param, room_size=1, decay=1.0, damping=0.0, Q_sig=q_format)
    elif algo == "mono_room":
        reverb = rv.reverb_room(fs, 1, max_room_size=param, room_size=1, decay=1.0, damping=0.0, Q_sig=q_format)
    elif algo == "stereo_plate":
        sig = np.tile(sig, [2, 1])
        reverb = rvp.reverb_plate_stereo(fs, 2, decay=param, damping=0.0, Q_sig=q_format)
    
    #print(reverb.get_buffer_lens())
    
    output_xcore = np.zeros_like(sig)
    output_flt = np.zeros_like(sig)

    if "stereo" in algo:
        for n in range(sig.shape[1]):
            output_xcore[:, n] = reverb.process_channels_xcore(sig[:, n])
        reverb.reset_state()
        for n in range(sig.shape[1]):
            output_flt[:, n] = reverb.process_channels(sig[:, n])
    else:
        for n in range(len(sig)):
            output_xcore[n] = reverb.process_xcore(sig[n])
        reverb.reset_state()
        for n in range(len(sig)):
            output_flt[n] = reverb.process(sig[n])


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


@pytest.mark.parametrize("max_room_size_diffusion", [0.5, 0.9])
@pytest.mark.parametrize("decay, damping", [[0.5, 0.35],
                                            [1.0, 0.0]])
@pytest.mark.parametrize("q_format", [27, 31])
@pytest.mark.parametrize("algo, width", [["mono_room", None],
                                         ["stereo_room", 1.0],
                                         ["stereo_plate", 1.0],]
                                         )
@pytest.mark.parametrize("wdmix", [0.5, 1.0])
def test_reverb_time(max_room_size_diffusion, decay, damping, q_format, width, algo, wdmix):
    # measure reverb time with chirp
    fs = 48000

    if "plate" in algo:
        pregain = 0.5**(q_format - 26)
    else:
        pregain = 0.015 * 2**(27 - q_format)

    sig = np.zeros(int(fs*max_room_size_diffusion*6) + fs)
    sig[:1*fs] = gen.log_chirp(fs, 1, 1, 20, 20000)
    sig = sig* (2**q_format - 1)/(2**q_format)

    if algo =="stereo_room":
        sig = np.tile(sig, [2, 1])
        reverb = rvs.reverb_room_stereo(fs, 2, max_room_size=max_room_size_diffusion, room_size=1, decay=decay, damping=damping, Q_sig=q_format, pregain=pregain, width=width)
    elif algo =="mono_room":
        reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size_diffusion, room_size=1, decay=decay, damping=damping, Q_sig=q_format, pregain=pregain)
    elif algo =="stereo_plate":
        sig = np.tile(sig, [2, 1])
        reverb = rvp.reverb_plate_stereo(fs, 2, early_diffusion=max_room_size_diffusion,
        late_diffusion=max_room_size_diffusion,
        decay=decay, damping=damping, Q_sig=q_format,
        pregain=pregain, width=width)
    reverb.set_wet_dry_mix(wdmix)

    output_xcore = np.zeros_like(sig)
    output_flt = np.zeros_like(sig)

    if "stereo" in algo:
        for n in range(sig.shape[1]):
            output_flt[:, n] = reverb.process_channels(sig[:, n])

        reverb.reset_state()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', utils.SaturationWarning)
            for n in range(sig.shape[1]):
                output_xcore[:, n] = reverb.process_channels_xcore(sig[:, n])

    else:
        for n in range(len(sig)):
            output_flt[n] = reverb.process(sig[n])

        reverb.reset_state()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', utils.SaturationWarning)
            for n in range(len(sig)):
                output_xcore[n] = reverb.process_xcore(sig[n])

    # if we triggered a saturation warning, can't guarantee arrays are the same
    sat_warn_flag = any([wi.category is utils.SaturationWarning for wi in w])

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
@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_noise_floor(max_room_size, decay, damping, algo):
    # check the reverb decays to 0 (no limit cycle noise)
    fs = 48000
    q_format = 27

    sig = np.zeros(int(fs*max_room_size*40) + fs)
    sig[:1*fs] = gen.log_chirp(fs, 1, 1, 20, 20000)
    sig = sig* (2**q_format - 1)/(2**q_format)

    if algo =="stereo_room":
        sig = np.tile(sig, [2, 1])
        reverb = rvs.reverb_room_stereo(fs, 2, max_room_size=max_room_size, room_size=1, decay=decay, damping=damping, Q_sig=q_format)
    elif algo =="mono_room":
        reverb = rv.reverb_room(fs, 1, max_room_size=max_room_size, room_size=1, decay=decay, damping=damping, Q_sig=q_format)
    elif algo == "stereo_plate":
        sig = np.tile(sig, [2, 1])
        reverb = rvp.reverb_plate_stereo(fs, 2, decay=decay, damping=damping, Q_sig=q_format)
    #print(reverb.get_buffer_lens())
    
    output_xcore = np.zeros_like(sig)
    output_flt = np.zeros_like(sig)

    if "stereo" in algo:
        for n in range(sig.shape[1]):
            output_flt[:, n] = reverb.process_channels(sig[:, n])

        reverb.reset_state()
        with warnings.catch_warnings(record=True) as w:
            for n in range(sig.shape[1]):
                output_xcore[:, n] = reverb.process_channels_xcore(sig[:, n])
    else:
        for n in range(len(sig)):
            output_flt[n] = reverb.process(sig[n])

        reverb.reset_state()
        with warnings.catch_warnings(record=True) as w:
            for n in range(len(sig)):
                output_xcore[n] = reverb.process_xcore(sig[n])

    # check noise floor
    if "stereo" in algo:
        assert np.max(np.abs(output_xcore[:, -1000:])) < 2**-(reverb.Q_sig + 1)
    else:
        assert np.max(np.abs(output_xcore[-1000:])) < 2**-(reverb.Q_sig + 1)

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = np.logical_and(utils.db(output_flt) > -50, utils.db(output_flt) < (6*(31-q_format)))
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_bypass(algo):
    # test that a drc component is bit exact when the signal is below
    # the threshold (or above in the case of a noise gate).
    fs = 48000
    signal = gen.log_chirp(fs, 0.5, 1)

    if algo == "stereo_room":
        signal = np.tile(signal, [2, 1])
        reverb = rvs.reverb_room_stereo(fs, 2, dry_gain_db=0, wet_gain_db=-np.inf)
    elif algo == "mono_room":
        reverb = rv.reverb_room(fs, 1, dry_gain_db=0, wet_gain_db=-np.inf)
    elif algo == "stereo_plate":
        signal = np.tile(signal, [2, 1])
        reverb = rvp.reverb_plate_stereo(fs, 2, dry_gain_db=0, wet_gain_db=-np.inf)
    
    output_xcore = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)

    if "stereo" in algo:
        for n in range(signal.shape[1]):
            output_xcore[:, n] = reverb.process_channels_xcore(signal[:, n])
        reverb.reset_state()
        for n in range(signal.shape[1]):
            output_flt[:, n] = reverb.process_channels(signal[:, n])
    else:
        for n in range(len(signal)):
            output_xcore[n] = reverb.process_xcore(signal[n])
        reverb.reset_state()
        for n in range(len(signal)):
            output_flt[n] = reverb.process(signal[n])

    np.testing.assert_array_equal(signal, output_flt)
    # quantization noise from multiply by dry gain
    np.testing.assert_allclose(signal, output_xcore, atol=2**-(reverb.Q_sig-1))

@pytest.mark.parametrize("algo", ["stereo_room", "stereo_plate"])
@pytest.mark.parametrize("width", [0, 1])
def test_reverb_width(algo, width):
    # test that a drc component is bit exact when the signal is below
    # the threshold (or above in the case of a noise gate).
    fs = 48000
    signal = gen.log_chirp(fs, 0.5, 1)

    if algo == "stereo_room":
        signal = np.tile(signal, [2, 1])
        reverb = rvs.reverb_room_stereo(fs, 2)
    elif algo == "mono_room":
        reverb = rv.reverb_room(fs, 1)
    elif algo == "stereo_plate":
        signal = np.tile(signal, [2, 1])
        reverb = rvp.reverb_plate_stereo(fs, 2)
    
    reverb.width = width

    output_xcore = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)

    if "stereo" in algo:
        for n in range(signal.shape[1]):
            output_xcore[:, n] = reverb.process_channels_xcore(signal[:, n])
        reverb.reset_state()
        for n in range(signal.shape[1]):
            output_flt[:, n] = reverb.process_channels(signal[:, n])
    else:
        for n in range(len(signal)):
            output_xcore[n] = reverb.process_xcore(signal[n])
        reverb.reset_state()
        for n in range(len(signal)):
            output_flt[n] = reverb.process(signal[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    q_format = 27
    top_half = np.logical_and(utils.db(output_flt) > -50, utils.db(output_flt) < (6*(31-q_format)))
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055

    if width == 0:
        assert np.all(output_xcore[0, :] == output_xcore)
        assert np.all(output_flt[0, :] == output_flt)
    else:
        assert not np.all(output_xcore[0, :] == output_xcore[1:, :])
        assert not np.all(output_flt[0, :] == output_flt[1:, :])


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("q_format", [27, 31])
@pytest.mark.parametrize("algo, param", [["mono_room", 0.1],
                                         ["mono_room", 1],
                                         ["mono_room", 4],
                                         ["stereo_room", 0.1],
                                         ["stereo_room", 1],
                                         ["stereo_room", 4],
                                         ["stereo_plate", 0.1],
                                         ["stereo_plate", 0.5],
                                         ["stereo_plate", 0.9],]
                                         )
def test_reverb_frames(fs, q_format, algo, param):
    # test the process_frame functions of the reverb components

    # if q_format > 27:
    #     pytest.xfail("This test is not meant to pass with a q more then 27")

    if algo == "stereo_room":
        reverb = rvs.reverb_room_stereo(fs, 2, max_room_size=param, width=0, Q_sig=q_format)

        signal = gen.log_chirp(fs, 0.5, 1)
        t = np.arange(len(signal))/fs
        signal *= np.sin(t*2*np.pi*0.5)
        signal = np.tile(signal, [2, 1])
    elif algo == "mono_room":
        reverb = rv.reverb_room(fs, 1, max_room_size=param, Q_sig=q_format)

        signal = gen.log_chirp(fs, 0.5, 1)
        t = np.arange(len(signal))/fs
        signal *= np.sin(t*2*np.pi*0.5)
        signal = np.tile(signal, [1, 1])
    elif algo == "stereo_plate":
        reverb = rvp.reverb_plate_stereo(fs, 2, decay=param, width=0, Q_sig=q_format)
    
        signal = gen.log_chirp(fs, 0.5, 1)
        t = np.arange(len(signal))/fs
        signal *= np.sin(t*2*np.pi*0.5)
        signal = np.tile(signal, [2, 1])

    frame_size = 1
    signal_frames = utils.frame_signal(signal, frame_size, 1)

    output_int = np.zeros_like(signal)
    output_flt = np.zeros_like(signal)

    for n in range(len(signal_frames)):
        output_int[:, n*frame_size:(n+1)*frame_size] = reverb.process_frame_xcore(signal_frames[n])
    reverb.reset_state()
    for n in range(len(signal_frames)):
        output_flt[:, n*frame_size:(n+1)*frame_size] = reverb.process_frame(signal_frames[n])

    assert np.all(output_int[0, :] == output_int)
    assert np.all(output_flt[0, :] == output_flt)


@pytest.mark.parametrize("ratio", [0, 0.5, 1])
@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_wet_dry_mix(ratio, algo):
    fs = 48000
    q_format = 27
    max_room_sz = 1
    room_sz = 1
    damp = 0.22

    a = utils.db2gain(-10)
    sig = gen.pink_noise(fs, 1, a)

    if algo == "stereo_room":
        sig = np.tile(sig, [2, 1])
        verb = rvs.reverb_room_stereo(fs, 2, max_room_size=max_room_sz, damping=damp, room_size=room_sz, Q_sig=q_format)
    elif algo =="mono_room":
        verb = rv.reverb_room(fs, 1, max_room_size=max_room_sz, damping=damp, room_size=room_sz, Q_sig=q_format)
    elif algo == "stereo_plate":
        sig = np.tile(sig, [2, 1])
        verb = rvp.reverb_plate_stereo(fs, 2, Q_sig=q_format)

    verb.set_wet_dry_mix(ratio)
    sig_py = np.zeros_like(sig)
    sig_xc = np.zeros_like(sig)
    if "stereo" in algo:
        for i in range(sig.shape[1]):
            sig_py[:, i] = verb.process_channels(sig[:, i])
        verb.reset_state()
        for i in range(sig.shape[1]):
            sig_xc[:, i] = verb.process_channels_xcore(sig[:, i])
    else:
        for i in range(len(sig)):
            sig_py[i] = verb.process(sig[i])
        verb.reset_state()
        for i in range(len(sig)):
            sig_xc[i] = verb.process_xcore(sig[i])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(sig_py) > -50
    if np.any(top_half):
        error_vpu = np.abs(utils.db(sig_py[top_half])-utils.db(sig_xc[top_half]))
        mean_error_vpu = utils.db(np.nanmean(utils.db2gain(error_vpu)))
        assert mean_error_vpu < 0.005

def get_algo_partial(algo):
    if algo =="stereo_room":
        r = partial(rvs.reverb_room_stereo, 48000, 2)
    elif algo =="mono_room":
        r = partial(rv.reverb_room, 48000, 1)
    elif algo =="stereo_plate":
        r = partial(rvp.reverb_plate_stereo, 48000, 2)
    
    return r

@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_properties_decay(algo):
    """Basic tests to check for consistency when setting the properties."""
    r = get_algo_partial(algo)

    val = 0.1
    a = r(decay=val)
    b = r()
    b.decay = val

    should_be_val = np.array([i.decay for i in (a, b)])
    np.testing.assert_allclose(should_be_val, val)


@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_properties_pregain(algo):
    """Basic tests to check for consistency when setting the properties."""
    r = get_algo_partial(algo)

    val = 0.1
    a = r(pregain=val)
    b = r()
    b.pregain = val

    should_be_val = np.array([i.pregain for i in (a, b)])
    np.testing.assert_allclose(should_be_val, val)


@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_properties_wet_db(algo):
    """Basic tests to check for consistency when setting the properties."""
    r = get_algo_partial(algo)

    val = -6
    a = r(wet_gain_db=val)
    b = r()
    b.wet_db = val

    should_be_val = np.array([i.wet_db for i in (a, b)])
    np.testing.assert_allclose(should_be_val, val)


@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_properties_dry_db(algo):
    """Basic tests to check for consistency when setting the properties."""
    r = get_algo_partial(algo)

    val = -6
    a = r(dry_gain_db=val)
    b = r()
    b.dry_db = val

    should_be_val = np.array([i.dry_db for i in (a, b)])
    np.testing.assert_allclose(should_be_val, val)


@pytest.mark.parametrize("algo", ["mono_room", "stereo_room", "stereo_plate"])
def test_reverb_properties_damping(algo):
    """Basic tests to check for consistency when setting the properties."""
    r = get_algo_partial(algo)

    val = 0.5
    a = r(damping=val)
    b = r()
    b.damping = val

    should_be_val = np.array([i.damping for i in (a, b)])
    np.testing.assert_allclose(should_be_val, val)

@pytest.mark.parametrize("stereo", [True, False])
def test_reverb_properties_room_size(stereo):
    """Basic tests to check for consistency when setting the properties."""
    if stereo:
        r = partial(rvs.reverb_room_stereo, 48000, 2)
    else:
        r = partial(rv.reverb_room, 48000, 1)

    val = 0.5
    a = r(room_size=val)
    b = r()
    b.room_size = val

    should_be_val = np.array([i.room_size for i in (a, b)])
    np.testing.assert_allclose(should_be_val, val)

if __name__ == "__main__":
    test_reverb_width("stereo_plate", 1)
    # test_reverb_time(0.5, 0.25, 0.35, 29, 0.5, 1, "stereo_plate")
    # test_reverb_overflow("sine", 20, "stereo_plate", 0.1)
    # test_reverb_time(0.01, 1)
    # test_reverb_frames(48000, 27, "stereo_plate", 0.5)
    # test_reverb_wet_dry_mix(1.0, "stereo_plate")
    # test_reverb_bypass_stereo()
    # test_reverb_noise_floor_stereo(1.0, 1.0, 0)
    # test_reverb_time(0.01, 1, 0, 31, 0.001, 0.5)
    # test_reverb_properties_room_size(True)
