.. _run_time_control_helper_section:

=================================
Run-Time Control Helper Functions
=================================

Most DSP Stages have fixed point control parameters. To aid conversion
from typical tuning units (e.g. decibels) to the correct fixed point
format, the helper functions below have been provided.


DRC helpers
===========

Calculate alpha
---------------

.. doxygenfunction:: calc_alpha

Calculate peak threshold
------------------------

.. doxygenfunction:: calculate_peak_threshold

Calculate RMS threshold
-----------------------

.. doxygenfunction:: calculate_rms_threshold

RMS compressor slope from ratio
-------------------------------

.. doxygenfunction:: rms_compressor_slope_from_ratio

Peak expander slope from ratio
------------------------------

.. doxygenfunction:: peak_expander_slope_from_ratio

