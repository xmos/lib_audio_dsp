.. _programming_guide_introduction:

INTRODUCTION
############

Lib audio DSP is a DSP library for the xcore. It also provides a DSP pipeline generation python library
for simple generation of mulithreaded audio DSP pipelines that efficiently utilise the xcore architecture.

This documentation separates the DSP library and pipeline generation into separate sections as it is
expected that some users will have custom use cases that require hand constructed DSP.

Adding lib_audio_dsp to your project
====================================

lib_audio_dsp has been designed to support xcommon cmake based projects. Therefore using the DSP part of this library
is as easy as adding "lib_audio_dsp" to your projects "APP_DEPENDENT_MODULES".

Using the pipeline generation utility will additionally require installing the python module that is found in the "python"
subdirectory of lib_audio_dsp. Having run cmake to download lib_audio_dsp into a sandbox, run the following command from the sandbox root::

    pip install -e lib_audio_dsp/python

It is important to re-run cmake after installing the python module so that the DSP design source files will be included in
the build. Read more on generating DSP pipelines in the :ref:`Pipeline Design API section<pipeline_design_api>`.


Signal Processing Performance
=============================

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

