app_simple_audio_dsp_integration
################################

This example shows the minimum required to integrate a DSP pipeline into an
xcore application. Two application threads are spawned, a "signal producer" and
a "signal consumer". The DSP pipeline is also initialised at the same time, and
will use as many threads as has been designed in `dsp_design.ipynb` - by default
one. The signal producer acts as a clock master, and every 48 kHz pushes a new
sample into the pipeline. This sample is drawn from a static array filled with
precomputed values to generate approximate white noise at 0 dBFS. The signal
consumer blocks on the output from the pipeline, and then places the processed
sample directly into an output buffer. When this buffer fills, a flag variable
is inverted and the output wraps around to the start of the buffer. By placing
a watchpoint in `xgdb` on this flag variable, the output buffer may be observed
at the point where it has been refilled. This is the minimum required to verify
the operation of the DSP pipeline with no physical audio interfaces.