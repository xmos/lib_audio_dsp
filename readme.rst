Lib Audio DSP
#############

lib_audio_dsp is a DSP library for the XMOS xcore architecture. It facilitates the creation of
multithreaded audio DSP pipelines that efficiently utilise the xcore architecture.

The library is built around a set of DSP function blocks, referred to in the documentation as “Stages” ,
which have a consistent API and can be combined to create many different designs. 

A tool for easily
combining Stages into a custom DSP pipeline is provided. DSP pipeline parameters can be adjusted and 
tuned on the fly via a PC based tuning interface, and utilities for hardware controls are also provided.

lib_audio_dsp includes common signal processing functions optimised for the xcore, such as:

* biquads and FIR filters
* compressors, limiters, noise gates and envelope detectors
* adders, subtractors, gains, volume controls and mixers
* delays and reverb.

See the :ref:`programming guide<programming_guide_introduction>` for usage details and getting started. 

Software version
****************

The CHANGELOG contains information about the current and previous versions.

Support
*******

This package is supported by XMOS Ltd. Issues can be raised against the software at: http://www.xmos.com/support

License
*******

This Software is subject to the terms of the `XMOS Public Licence: Version 1 <https://github.com/xmos/lib_audio_dsp/LICENSE.rst>`_.
