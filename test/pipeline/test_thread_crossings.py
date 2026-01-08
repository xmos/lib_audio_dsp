# Copyright 2025-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages import *


def test_bad_pipelines():
    with pytest.raises(RuntimeError) as excinfo:
        p, inputs = Pipeline.begin(2, fs=48000)

        # inputs[1] is not used on thread 0
        x1 = p.stage(Biquad, inputs[0], 'bq1')

        p.next_thread()

        # inputs[1] first used on thread 1
        x = p.stage(Biquad, x1 + inputs[1], 'bq2')

        p.set_outputs(x)
    
    assert "must cross the same number of threads" in str(excinfo.value)


def test_good_pipelines():

    p, inputs = Pipeline.begin(2, fs=48000)

    # both inputs are not used on this thread
    x1 = p.stage(Biquad, inputs[0], 'bq1')
    x2 = p.stage(Bypass, inputs[1], 'by1')

    p.next_thread()

    x = p.stage(Biquad, x1 + x2, 'bq2')

    p.set_outputs(x)


def test_parallel_pipelines():

    p, inputs = Pipeline.begin(2, fs=48000)

    x1 = p.stage(Biquad, inputs[0], 'bq1')
    p.next_thread()

    x2 = p.stage(Bypass, inputs[1], 'by1')
    p.next_thread()

    # x1 and x2 have both crossed 1 thread already
    x = p.stage(Biquad, x1 + x2, 'bq2')

    p.set_outputs(x)


def test_good_complex():

    # 4 inputs
    p, i = Pipeline.begin(4, fs=48000)

    mic1 = p.stage(Fork, i[2], count=3).forks
    mic2 = p.stage(Fork, i[3], count=3).forks

    p.stage(EnvelopeDetectorRMS, mic1[0], "edr1")
    p.stage(EnvelopeDetectorRMS, mic2[0], "edr2")

    mic_mono = p.stage(Adder, mic1[1] + mic2[1], "adder1")
    mic_mono = p.stage(Fork, mic_mono, count=3).forks
    mic_dnr = p.stage(NoiseSuppressorExpander, mic_mono[0], "ns")
    peq1 = p.stage(CascadedBiquads, mic_dnr, "peq1")
    sw_1 = p.stage(SwitchStereo, mic_mono[1] + mic_mono[2] + mic1[2] + mic2[2], "sw_1")
    i_bypass = p.stage(Bypass, i[:2])

    p.next_thread()
    mic_peq = p.stage(Fork, peq1, count=3).forks
    peq2 = p.stage(CascadedBiquads, mic_peq[2], "peq2")
    mic_rv = p.stage(Fork, peq2).forks
    i_bypass = p.stage(Bypass, i_bypass)
    sw_1 = p.stage(Bypass, sw_1)

    p.next_thread()
    mic_rv = p.stage(ReverbPlateStereo, mic_rv[0] + mic_rv[1], "reverb")
    cfs1 = p.stage(CrossfaderStereo, mic_peq[0] + mic_peq[1] + mic_rv[0] + mic_rv[1], "cfs1")
    rv_sw = p.stage(SwitchStereo, cfs1 + sw_1, "reverb_sw")
    vol1 = p.stage(VolumeControl, rv_sw[0], "vol1")
    vol2 = p.stage(VolumeControl, rv_sw[1], "vol2")
    i_bypass = p.stage(Bypass, i_bypass)

    p.next_thread()
    usb_play, usb_mon = p.stage(Fork, i_bypass).forks
    usb_play = p.stage(VolumeControl, usb_play, "vol3")
    mic_play, mic_mon = p.stage(Fork, vol1 + vol2).forks
    usb0 = p.stage(Adder, usb_play[0] + mic_play[0], "adder2")
    usb1 = p.stage(Adder, usb_play[1] + mic_play[1], "adder3")
    cfs2 = p.stage(CrossfaderStereo, usb_mon[0] + usb_mon[1] + mic_mon[0] + mic_mon[1], "cfs2")
    mon0, mon1 = p.stage(VolumeControl, cfs2, "vol4")

    p.set_outputs(usb0 + usb1 + mon0 + mon1)


def test_bad_complex():
    with pytest.raises(RuntimeError) as excinfo:
        # 4 inputs
        p, i = Pipeline.begin(4, fs=48000)

        mic1 = p.stage(Fork, i[2], count=3).forks
        mic2 = p.stage(Fork, i[3], count=3).forks

        p.stage(EnvelopeDetectorRMS, mic1[0], "edr1")
        p.stage(EnvelopeDetectorRMS, mic2[0], "edr2")

        mic_mono = p.stage(Adder, mic1[1] + mic2[1], "adder1")
        mic_mono = p.stage(Fork, mic_mono, count=3).forks
        mic_dnr = p.stage(NoiseSuppressorExpander, mic_mono[0], "ns")
        peq1 = p.stage(CascadedBiquads, mic_dnr, "peq1")
        sw_1 = p.stage(SwitchStereo, mic_mono[1] + mic_mono[2] + mic1[2] + mic2[2], "sw_1")

        p.next_thread()
        mic_peq = p.stage(Fork, peq1, count=3).forks
        peq2 = p.stage(CascadedBiquads, mic_peq[2], "peq2")
        mic_rv = p.stage(Fork, peq2).forks
        
        p.next_thread()
        mic_rv = p.stage(ReverbPlateStereo, mic_rv[0] + mic_rv[1], "reverb")
        cfs1 = p.stage(CrossfaderStereo, mic_peq[0] + mic_peq[1] + mic_rv[0] + mic_rv[1], "cfs1")
        rv_sw = p.stage(SwitchStereo, cfs1 + sw_1, "reverb_sw")
        vol1 = p.stage(VolumeControl, rv_sw[0], "vol1")
        vol2 = p.stage(VolumeControl, rv_sw[1], "vol2")

        p.next_thread()
        usb_play, usb_mon = p.stage(Fork, i[:2]).forks
        usb_play = p.stage(VolumeControl, usb_play, "vol3")
        mic_play, mic_mon = p.stage(Fork, vol1 + vol2).forks
        usb0 = p.stage(Adder, usb_play[0] + mic_play[0], "adder2")
        usb1 = p.stage(Adder, usb_play[1] + mic_play[1], "adder3")
        cfs2 = p.stage(CrossfaderStereo, usb_mon[0] + usb_mon[1] + mic_mon[0] + mic_mon[1], "cfs2")
        mon0, mon1 = p.stage(VolumeControl, cfs2, "vol4")

        p.set_outputs(usb0 + usb1 + mon0 + mon1)

    assert "must cross the same number of threads" in str(excinfo.value)

if __name__ == "__main__":
    # test_bad_pipelines()
    test_bad_complex()
    test_good_complex()