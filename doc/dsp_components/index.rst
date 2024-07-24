##############
DSP Components
##############

.. rubric:: Introduction

lib_audio_dsp provides many common signal processing function optimised
for xcore. These can be combined together to make complex audio pipelines
for many different applications, such as home audio, music production,
voice processing and AI feature extraction.

The library is split into 2 levels of API, DSP Stages and DSP Modules.
Both APIs provide similar DSP functionality, but are suited to different
use cases.

.. rubric:: DSP Stages

The higher-level APIs are called *DSP Stages*. These stages are designed
to work with the Python DSP pipeline tool. This tool allows developers
to quickly and easily create, test, and deploy DSP pipelines without
needing to write a lot of code. By using DSP Stages, the user can build
complex audio processing workflows in a short amount of time, making it
ideal for rapid prototyping and development.

.. rubric:: DSP Modules

The lower-level APIs are called *DSP Modules*. Those are meant to be
used as an API directly in cases where the Python DSP pipeline tool is 
not used. These can be useful when integrating DSP function into an
existing system, or as a starting point for creating bespoke DSP
functions.


.. rubric:: Precision

.. note::
    For fixed point Q formats this document uses the format QM.N, where M is the number of bits
    before the decimal point (excluding the sign bit), and N is the number of bits after the decimal
    point. For an int32 number, M+N=31.

By default, the signal processing in the audio pipeline is carried out at 32 bit fixed point
precision in Q4.27 format. Assuming a 24 bit input signal in Q0.24 format, this gives 4 bits of
internal headroom in the audio pipeline. The output of the audio pipeline will be clipped back to
Q0.24 before returning. For more precision, the pipeline can be configured to run with no headroom
in Q0.31 format, but this requires manual headroom management.

DSP algorithms are implemented either on the XS3 CPU or VPU (vector processing unit).

CPU algorithms are typically implemented as 32-bit x 32-bit operations into 64-bit results and
accumulators, before rounding back to 32-bit outputs.

The VPU allows for 8 simultaneous operations, with a small cost in precision. VPU algorithms are
typically implemented as 32-bit x 32-bit operations into 34-bit results and 40-bit accumulators,
before rounding back to 32-bit outputs.


.. rubric:: Latency

The latency of the DSP pipeline is dependent on the number of threads. By default, the DSP pipeline
is configured for one sample of latency per thread. All current DSP modules have zero inbuilt
latency (except where specified e.g. delays stages). For pipelines that fit on a single thread,
this means the total pipeline latency is 1 sample.

The pipeline can also be configured to use a higher frame size. This increases latency, but can
reduce compute for simple functions. For a pipeline consisting of just biquads:

* Frame size = 1, latency = 1 sample, compute = 25 biquads per thread @ 48kHz.
* Frame size = 8, latency = 8 samples, compute = 60 biquads per thread @ 48kHz.


.. toctree::
    :maxdepth: 1

    modules/index.rst
    stages/index.rst
    runtime_control/index.rst
