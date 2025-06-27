Latency
=======

The latency of the DSP pipeline is dependent on the number of threads. By default, the DSP pipeline
is configured for one sample of latency per thread. All current DSP modules have zero inbuilt
latency (except where specified e.g. delay stages). For pipelines that fit on a single thread,
this means the total pipeline latency is 1 sample.

The pipeline can also be configured to use a higher frame size. This increases latency, but can
reduce compute for simple functions, as the function overhead is shared. For a pipeline consisting of just biquads:


.. table::
  :widths: 15, 15, 15, 15, 15

  ==========  =======  ==================  ===================  ======================
  Frame size  Latency  % thread per frame  % thread per sample  Max biquads per thread
  ==========  =======  ==================  ===================  ======================
  1           1        4.0%                4.0%                 25
  8           8        13.3%               1.7%                 60
  ==========  =======  ==================  ===================  ======================
