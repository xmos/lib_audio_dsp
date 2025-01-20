.. _run_time_control_helper_section:

========================
Control Helper Functions
========================

Most DSP Stages have fixed point control parameters. To aid conversion
from typical tuning units (e.g. decibels) to the correct fixed point
format, the helper functions below have been provided.


Biquad helpers
==============

.. doxygenfile:: control/biquad.h


DRC helpers
===========

.. doxygenfunction:: calc_alpha

.. doxygenfunction:: calculate_peak_threshold

.. doxygenfunction:: calculate_rms_threshold

.. doxygenfunction:: rms_compressor_slope_from_ratio

.. doxygenfunction:: peak_expander_slope_from_ratio

Reverb helpers
==============

.. doxygenfile:: control/reverb.h


Signal chain helpers
====================

.. doxygenfunction:: time_to_samples
