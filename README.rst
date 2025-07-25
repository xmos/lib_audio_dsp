:orphan:

#############################################
lib_audio_dsp: Audio DSP Library for xcore.ai
#############################################

:vendor: XMOS
:version: 1.4.1alpha0
:scope: General Use
:description: Audio DSP Library for xcore.ai
:category: Audio
:keywords: library, DSP, Audio, Audio Effects, Audio Pipeline
:devices: xcore.ai

*******
Summary
*******

``lib_audio_dsp``

.. note::

  Some software components in this tool flow are prototypes and will be updated in Version 2 of the library.
  The underlying Digital Signal Processing (DSP) blocks are however fully functional. Future updates will
  enhance the features and flexibility of the design tool.

lib_audio_dsp is a DSP library for the XMOS xcore architecture. It facilitates the creation of
multithreaded audio DSP pipelines that efficiently utilise the xcore architecture.

The library is built around a set of DSP function blocks, referred to in the documentation as "Stages",
which have a consistent API and can be combined to create many different designs.

********
Features
********

lib_audio_dsp includes common signal processing functions optimised for the xcore, such as:

* biquads and FIR filters
* compressors, limiters, noise gates and envelope detectors
* adders, subtractors, gains, volume controls and mixers
* delays and reverb.

These can be combined together to make complex audio pipelines for many
different applications, such as home audio, music production, voice
processing, and AI feature extraction.

************
Known issues
************

  * None


****************
Development repo
****************

  * `lib_audio_dsp <https://www.github.com/xmos/lib_audio_dsp>`_

**************
Required tools
**************

  * XMOS XTC Tools: 15.3.1
  
************************
Host system requirements
************************

  * None

*********************************
Required libraries (dependencies)
*********************************

  * lib_xcore_math (www.github.com/xmos/lib_xcore_math)
  * lib_logging (www.github.com/xmos/lib_logging)
  * lib_locks (www.github.com/xmos/lib_locks)

*************************
Related application notes
*************************

The following application notes use this library:

  * `AN02014: Integrating DSP Into The XMOS USB Reference Design <https://www.xmos.com/file/an02014-integrating-dsp-into-the-xmos-usb-reference-design/>`_.
  * `AN02015: Run-time DSP control in a USB Audio Application <https://www.xmos.com/file/an02015-run-time-dsp-control-in-a-usb-audio-application/>`_.

*******
Support
*******

This package is supported by XMOS Ltd. Issues can be raised against the software at: http://www.xmos.com/support
