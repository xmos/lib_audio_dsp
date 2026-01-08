# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.


import numpy as np
from python import build_utils, run_pipeline_xcoreai, audio_helpers
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages.biquad import Biquad

def test_pipeline_executor():
    p, i = Pipeline.begin(4)
    n_samps = 100
    s = p.stage(Biquad, i)
    s = p.stage(Biquad, s)
    s0 = p.stage(Biquad, s[:2])
    s1 = p.stage(Biquad, s[2:])
    s = p.stage(Biquad, s0 + s1)
    p.set_outputs(s)

    executor = p.executor()
    sig = np.ones((n_samps, 4))

    ret = executor.process(sig).data
    print(ret)

