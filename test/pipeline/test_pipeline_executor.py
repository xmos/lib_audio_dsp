

import numpy as np
from python import build_utils, run_pipeline_xcoreai, audio_helpers
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages.biquad import Biquad

def test_pipeline_executor():
    p = Pipeline(4)
    t = p.add_thread()
    n_samps = 100
    s = t.stage(Biquad, p.i)
    s = t.stage(Biquad, s.o)
    s0 = t.stage(Biquad, s.o[:2])
    s1 = t.stage(Biquad, s.o[2:])
    s = t.stage(Biquad, s0.o + s1.o)
    p.set_outputs(s.o)

    executor = p.executor()
    sig = np.ones((n_samps, 4))

    ret = executor.process(sig).data
    print(ret)

