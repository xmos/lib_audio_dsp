import numpy as np
import pytest

import audio_dsp.dsp.drc as drc
import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_gen as gen


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("at", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("threshold", [-20, -10, -6, 0])
def test_limiter_peak_attack(fs, at, threshold):
    # Attack time test bads on Figure 2 in Guy McNally's "Dynamic Range Control
    # of Digital Audio Signals"
    x = np.ones(int(at*2*fs))
    x = utils.db2gain(threshold + 6)
    t = np.arange(len(x))/fs

    lt = drc.limiter_peak(fs, threshold, at, 0.3)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

    # find when within 3dB of threshold
    thresh_passed = np.argmax(utils.db(env) > threshold)
    sig_3dB = np.argmax(utils.db(y) < threshold + 2)

    measured_at = t[sig_3dB] - t[thresh_passed]
    print("target: %.3f, measured: %.3f" % (at, measured_at))

    # be somewhere vaugely near the spec, attack time definition is variable!
    assert measured_at/at > 0.85
    assert measured_at/at < 1.15


@pytest.mark.parametrize("fs", [48000])
@pytest.mark.parametrize("rt", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("threshold", [-20, -10, -6, 0])
def test_limiter_peak_release(fs, rt, threshold):
    # Release time test bads on Figure 2 in Guy McNally's "Dynamic Range
    # Control of Digital Audio Signals"
    x = np.ones(int((0.5+rt*2)*fs))
    x[:fs//2] = utils.db2gain(threshold + 6)
    x[fs//2:] = utils.db2gain(threshold - 3)
    t = np.arange(len(x))/fs

    lt = drc.limiter_peak(fs, threshold, 0.01, rt)

    y = np.zeros_like(x)
    f = np.zeros_like(x)
    env = np.zeros_like(x)

    for n in range(len(y)):
        y[n], f[n], env[n] = lt.process(x[n])

    # find when within 3dB of threshold
    sig_3dB = np.argmax(utils.db(y[fs//2:]) > threshold - 2 - 3)

    measured_rt = t[sig_3dB]
    print("target: %.3f, measured: %.3f" % (rt, measured_rt))
    print(measured_rt/rt)

    # be somewhere vaugely near the spec, attack time definition is variable!
    assert measured_rt/rt > 0.8
    assert measured_rt/rt < 1.2


# TODO RMS tests
# TODO hard limiter test
# TODO int vs float test
# TODO envelope detector tests
# TODO compressor tests
