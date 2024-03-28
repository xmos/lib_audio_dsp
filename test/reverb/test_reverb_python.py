# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np
import pytest

import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_gen as sg
import audio_dsp.dsp.generic as dspg
import audio_dsp.dsp.reverb as rv

@pytest.mark.parametrize("freq", [10, 100, 1000, 10000, 23000])
@pytest.mark.parametrize("max_room_size", [0.1, 0.5, 1, 2, 4])
def test_reverb_overflow(freq, max_room_size):
    fs = 48000

    sig = sg.sin(fs, 5, freq, 1)
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


if __name__ == "__main__":
    test_reverb_overflow(1000)
