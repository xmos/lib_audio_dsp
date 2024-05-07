# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.generic as dspg
from audio_dsp.dsp.generic import HEADROOM_DB

import soundfile as sf


def chirp_filter_test(filter, fs):
    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)

    output_flt = np.zeros(len(signal))
    output_xcore = np.zeros(len(signal))

    for n in np.arange(len(signal)):
        output_flt[n] = filter.process(signal[n])
    for n in np.arange(len(signal)):
        output_xcore[n] = filter.process_xcore(signal[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_vpu = np.abs(utils.db(output_flt[top_half])-utils.db(output_xcore[top_half]))
        mean_error_vpu = utils.db(np.nanmean(utils.db2gain(error_vpu)))
        assert mean_error_vpu < 0.05


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("filter_n", np.arange(4))
@pytest.mark.parametrize("n_chans", [1])
def test_gains(filter_n, fs, n_chans):
    filter_spec = [['fixed_gain', -10],
                   ['fixed_gain', 24],
                   ['volume_control', 24],
                   ['volume_control', -10]]

    filter_spec = filter_spec[filter_n]

    class_name = f"{filter_spec[0]}"
    class_handle = getattr(sc, class_name)
    filter = class_handle(fs, n_chans, *filter_spec[1:])

    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("filter_n", np.arange(4))
@pytest.mark.parametrize("n_chans", [1, 2, 4])
def test_gains_frames(filter_n, fs, n_chans):
    filter_spec = [['fixed_gain', -10],
                   ['fixed_gain', 24],
                   ['volume_control', 24],
                   ['volume_control', -10]]

    filter_spec = filter_spec[filter_n]

    class_name = f"{filter_spec[0]}"
    class_handle = getattr(sc, class_name)
    filter = class_handle(fs, n_chans, *filter_spec[1:])

    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_flt = np.zeros_like(signal)
    output_xcore = np.zeros_like(signal)
    frame_size = 1

    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = filter.process_frame(signal_frames[n])
    assert np.all(output_flt[0, :] == output_flt)

    for n in range(len(signal_frames)):
        output_xcore[:, n:n+frame_size] = filter.process_frame_xcore(signal_frames[n])
    assert np.all(output_xcore[0, :] == output_xcore)


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("filter_spec", [['mixer', 2, 0],
                                         ['adder', 2],
                                         ['subtractor', 2]])
def test_saturation(filter_spec, fs):

    class_name = f"{filter_spec[0]}"
    class_handle = getattr(sc, class_name)
    if filter_spec[0] == "subtractor":
        # subtractor has fewer inputs
        filter = class_handle(fs)
    else:
        filter = class_handle(fs, *filter_spec[1:])

    length = 0.05
    signals = []
    for n in range(filter_spec[1]):
        # max level is 24db (16), so 10 + 10 will saturate
        signals.append(gen.sin(fs, length, 997, 10.0))
    if class_name == "subtractor":
        signals[1] *= -1
    signal = np.stack(signals, axis=0)
    signal = utils.saturate_float_array(signal, dspg.Q_SIG)
    
    output_flt = np.zeros(signal.shape[1])
    output_xcore = np.zeros(signal.shape[1])

    for n in range(signal.shape[1]):
        output_flt[n] = filter.process_channels(signal[:, n])

    for n in range(signal.shape[1]):
        output_xcore[n] = filter.process_channels_xcore(signal[:, n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    # also, int signals can't go above HEADROOM_BITS
    top_half = (utils.db(output_flt) > -50) * (utils.db(output_flt) < HEADROOM_DB)
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055
    assert np.all(utils.db(output_xcore) <= HEADROOM_DB)


def test_volume_change():
    fs = 48000
    start_gain = -60
    filter = sc.volume_control(fs, 1, start_gain)
    length = 5
    signal = gen.sin(fs, length, 997/2, 0.5)
    signal += gen.sin(fs, length, 997, 0.5)

    output_flt = np.zeros(len(signal))
    output_xcore = np.zeros(len(signal))

    steps = 12
    for step in range(steps):
        start = step*len(signal)//steps
        for n in range(len(signal)//steps):
            output_flt[start + n] = filter.process(signal[start + n])
        for n in range(len(signal)//steps):
            output_xcore[start + n] = filter.process_xcore(signal[start + n])
        filter.set_gain(start_gain + 6*step)

    # this is a very useful signal to listen to for clicks
    sf.write("vol_test_output_flt_slew.wav", output_flt, fs)

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


def test_mute():
    fs = 48000
    start_gain = -10
    filter = sc.volume_control(fs, 1, start_gain)
    length = 5
    signal = gen.sin(fs, length, 997/2, 0.5)
    signal += gen.sin(fs, length, 997, 0.5)

    output_flt = np.zeros(len(signal))
    output_xcore = np.zeros(len(signal))

    # check muting while muted, unmuting while unmuted, gain change while muted
    step_states = ["gain", "mute", "unmute", "mute", "mute", "unmute", "unmute", "mute", "gain", "unmute"]
    steps = len(step_states)
    for step in range(len(step_states)):
        if step_states[step] == "gain":
            start_gain += 3
            filter.set_gain(start_gain)
        elif step_states[step] == "mute":
            filter.mute()
        elif step_states[step] == "unmute":
            filter.unmute()

        start = step*len(signal)//steps
        for n in range(len(signal)//steps):
            output_flt[start + n] = filter.process(signal[start + n])
        for n in range(len(signal)//steps):
            output_xcore[start + n] = filter.process_xcore(signal[start + n])


    sf.write("mute_test_output_flt_slew.wav", output_flt, fs)

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("filter_spec", [['mixer', 2, 0],
                                         ['mixer', 3, -9],
                                         ['mixer', 2, -6],
                                         ['mixer', 4, -12],
                                         ['adder', 2],
                                         ['adder', 4],
                                         ['subtractor', 2]])
def test_combiners(filter_spec, fs):

    class_name = f"{filter_spec[0]}"
    class_handle = getattr(sc, class_name)
    if filter_spec[0] == "subtractor":
        # subtractor has fewer inputs
        filter = class_handle(fs)
    else:
        filter = class_handle(fs, *filter_spec[1:])

    length = 0.05
    signals = []
    for n in range(filter_spec[1]):
        signals.append(gen.pink_noise(fs, length, 1.0))
    signal = np.stack(signals, axis=0)
    signal = utils.saturate_float_array(signal, dspg.Q_SIG)

    output_flt = np.zeros(signal.shape[1])
    output_xcore = np.zeros(signal.shape[1])

    for n in range(signal.shape[1]):
        output_flt[n] = filter.process_channels(signal[:, n])

    for n in range(signal.shape[1]):
        output_xcore[n] = filter.process_channels_xcore(signal[:, n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("filter_spec", [['mixer', 2, 0],
                                         ['mixer', 3, -9],
                                         ['mixer', 2, -6],
                                         ['mixer', 4, -12],
                                         ['adder', 2],
                                         ['adder', 4],
                                         ['subtractor', 2]])
def test_combiners_frames(filter_spec, fs):

    class_name = f"{filter_spec[0]}"
    class_handle = getattr(sc, class_name)

    if filter_spec[0] == "subtractor":
        # subtractor has fewer inputs
        filter = class_handle(fs)
    else:
        filter = class_handle(fs, *filter_spec[1:])

    length = 0.05
    signals = []
    for n in range(filter_spec[1]):
        signals.append(gen.pink_noise(fs, length, 1.0))
    signal = np.stack(signals, axis=0)
    signal = utils.saturate_float_array(signal, dspg.Q_SIG)
    signal_frames = utils.frame_signal(signal, 1, 1)

    output_flt = np.zeros((1, len(signal)))
    output_xcore = np.zeros((1, len(signal)))
    frame_size = 1

    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = filter.process_frame(signal_frames[n])

    for n in range(len(signal_frames)):
        output_xcore[:, n:n+frame_size] = filter.process_frame_xcore(signal_frames[n])

    # small signals are always going to be ropey due to quantizing, so just check average error of top half
    top_half = utils.db(output_flt) > -50
    if np.any(top_half):
        error_flt = np.abs(utils.db(output_xcore[top_half])-utils.db(output_flt[top_half]))
        mean_error_flt = utils.db(np.nanmean(utils.db2gain(error_flt)))
        assert mean_error_flt < 0.055

@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("delay_spec", [[15, 10, "samples"],
                                        [128, 128, "samples"],
                                        [2, 1.7, "ms"],
                                        [1.056, 0.94, "s"]])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
def test_delay(fs, delay_spec, n_chans):
    filter = sc.delay(fs, n_chans, *delay_spec)

    delay_samps = filter._get_delay_samples(delay_spec[1], delay_spec[2])

    length = 0.005
    sig_len = int(length * fs)
    signal = gen.pink_noise(fs, length, 0.5)
    signal = np.pad(signal, (0, delay_samps))
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_flt = np.zeros_like(signal)
    output_xcore = np.zeros_like(signal)
    frame_size = 1

    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = filter.process_frame(signal_frames[n])
    assert np.all(signal[:, : sig_len] == output_flt[:, delay_samps :])

    for n in range(len(signal_frames)):
        output_xcore[:, n:n+frame_size] = filter.process_frame_xcore(signal_frames[n])
    assert np.all(signal[:, : sig_len] == output_xcore[:, delay_samps :])


if __name__ == "__main__":
    test_combiners(["adder", 4], 48000)
    # test_volume_change()
    # test_gains(1, 48000, 1)
