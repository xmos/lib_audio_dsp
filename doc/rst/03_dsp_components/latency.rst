Latency
=======

The latency of the DSP pipeline is dependent on the number of threads. By default, the DSP pipeline
is configured for one sample of latency per thread. All current DSP modules have zero inbuilt
latency (except where specified e.g. delay stages). For pipelines that fit on a single thread,
this means the total pipeline latency is 1 sample.

The pipeline can also be configured to use a higher frame size. This increases latency, but can
reduce compute for simple functions. For a pipeline consisting of just biquads:

* Frame size = 1, latency = 1 sample, compute = 25 biquads per thread @ 48kHz.
* Frame size = 8, latency = 8 samples, compute = 60 biquads per thread @ 48kHz.
