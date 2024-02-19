import pytest
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.signal_gen as gen
import audio_dsp.dsp.utils as utils


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
        signals.append(gen.pink_noise(fs, length, 0.5))
    signal = np.stack(signals, axis=0)

    output_flt = np.zeros(signal.shape[1])
    output_xcore = np.zeros(signal.shape[1])

    for n in range(signal.shape[1]):
        output_flt[n] = filter.process(signal[:, n])

    for n in range(signal.shape[1]):
        output_xcore[n] = filter.process_xcore(signal[:, n])

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
        signals.append(gen.pink_noise(fs, length, 0.5))
    signal = np.stack(signals, axis=0)

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_flt = np.zeros((1, len(signal)))
    output_xcore = np.zeros((1, len(signal)))
    frame_size = 1

    for n in range(len(signal_frames)):
        output_flt[:, n:n+frame_size] = filter.process_frame(signal_frames[n])

    for n in range(len(signal_frames)):
        output_xcore[:, n:n+frame_size] = filter.process_frame_xcore(signal_frames[n])

    # TODO add a test here? flt vpu similarity already tested in test_combiners


if __name__ == "__main__":
    # test_combiners(["subtractor", 2], 48000)
    test_gains(1, 48000, 1)
