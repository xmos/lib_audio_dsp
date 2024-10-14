Audio DSP library change log
============================

Unreleased
----------

  * ADDED: Stereo reverb Stage.
  * FIXED: Delay could return different type to input.
  * CHANGED: Removed host app.
  * ADDED: Tuning API added to Python library.

1.1.0
-----

  * CHANGED: Moved control related DSP from api/dsp to api/control.
  * CHANGED: Increased reverb wet signal level by 10 dB.
  * CHANGED: Added pre-delay to reverb.
  * FIXED: Removed configuration parameters from reverb that did not support being changed.
  * ADDED: Biquad design APIs for common filters.
  * ADDED: Helper functions for converting control parameters to fixed point formats.
  * ADDED: Improved documentation on control parameters.
  * ADDED: Wet-dry mix helper for reverb.
  * ADDED: Use Python properties for controllable parameters.
  * ADDED: Separate Python base classes for peak and RMS DRC components.

1.0.0
-----

  * ADDED: User guides for the Python library and DSP library in this repository

0.3.0
-----

  * CHANGED: Pipeline design API breaking change to simplify pipeline definitions.
  * CHANGED: Control API breaking change to support multiple control threads.
  * CHANGED: XCommon CMake version changed to v1.0.0.
  * CHANGED: Host app moved into this repository.
  * CHANGED: Improved memory allocation and macros.
  * CHANGED: Noise suppressor stage renamed to noise suppressor expander.
  * CHANGED: Reverb stage renamed to reverb room.
  * CHANGED: Faster envelope detector implementation.
  * CHANGED: DSP components saturate instead of overflowing.
  * ADDED: Full control of reverb component parameters 
  * ADDED: FIR filter DSP components and stages.
  * ADDED: Hard peak limiter and clipper DSP components and stages.
  * ADDED: Delay line DSP component and stage.
  * ADDED: C and Python documentation.

0.2.0
-----

  * ADDED: Pipeline checksum.
  * ADDED: Pipeline simulator.
  * ADDED: Basic performance visualization.
  * ADDED: Support for user-defined stage labels.
  * ADDED: Envelope detector stages.
  * ADDED: RMS compressor DSP components and stages.
  * ADDED: Noise gate and noise suppressor DSP components and stages.
  * ADDED: Signal chain DSP components and stages.

0.1.0
-----

  * Initial release.

