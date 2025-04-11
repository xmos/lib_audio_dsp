# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import numpy as np
import audio_dsp.dsp.graphic_eq as geq
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
@pytest.mark.parametrize("n_chans", [1, 2])
@pytest.mark.parametrize("gains", [[0, -12, 0, 12, 0, -12, 0, 12, 0, -12],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def test_geq(gains, fs, n_chans):

    filter = geq.graphic_eq_10_band(fs, n_chans, gains)
    chirp_filter_test(filter, fs)


@pytest.mark.parametrize("fs", [48000, 96000, 192000])
@pytest.mark.parametrize("n_chans", [1, 2, 4])
@pytest.mark.parametrize("gains", [[0, -12, 0, 12, 0, -12, 0, 12, 0, -12],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
@pytest.mark.parametrize("q_format", [27, 31])
def test_geq_frames(gains, fs, n_chans, q_format):

    filter = geq.graphic_eq_10_band(fs, n_chans, gains, Q_sig=q_format)

    length = 0.05
    signal = gen.log_chirp(fs, length, 0.5)
    signal = np.tile(signal, [n_chans, 1])

    signal_frames = utils.frame_signal(signal, 1, 1)

    output_flt = np.zeros_like(signal)
    output_xcore = np.zeros_like(signal)
    frame_size = 1

    for n in range(len(signal_frames)):
        output_flt[:, n*frame_size:(n+1)*frame_size] = filter.process_frame(signal_frames[n])
    assert np.all(output_flt[0, :] == output_flt)

    for n in range(len(signal_frames)):
        output_xcore[:, n*frame_size:(n+1)*frame_size] = filter.process_frame_xcore(signal_frames[n])
    assert np.all(output_xcore[0, :] == output_xcore)

if __name__ == "__main__":
    # test_geq([0, -12, 0, 12, 0, -12, 0, 12, 0, -12], 48000, 1)
    test_geq([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 48000, 1)

