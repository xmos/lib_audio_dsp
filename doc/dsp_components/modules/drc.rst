#####################
Dynamic Range Control
#####################

This page contains modules that provide the dynamic adjustment according to the range of the signal.

==================
Envelope Detectors
==================

Envelope detectors run an exponential moving avarage (EMA) of the incoming signal. They are used as a part of
the most DRC components. Can also be used to implement the UV meters.

.. doxygenstruct:: env_detector_t
    :members:

.. _env_det_peak:

----------------------
Envelope Detector Peak
----------------------

Peak-based envelope detector will run it's EMA using the absolute value of the input sample.

.. doxygenfunction:: adsp_env_detector_peak

.. autoclass:: audio_dsp.dsp.drc.drc.envelope_detector_peak
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

.. _env_det_rms:

---------------------
Envelope Detector RMS
---------------------

RMS-based envelope detector will run it's EMA using the square of the input sample.

.. doxygenfunction:: adsp_env_detector_rms

.. autoclass:: audio_dsp.dsp.drc.drc.envelope_detector_rms
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

==================================
Clippers, Limiters and Compressors
==================================

Limiters and compressors attenuate the signals that are above the given threshold. Compressor
level above the threshold is defined by the `ratio` parameter. Limiter acts as a compressor with
an infinite `ratio`.

-------
Clipper
-------

Will clip an input value if it's above the threshold.

.. doxygentypedef:: clipper_t

.. doxygenfunction:: adsp_clipper

.. autoclass:: audio_dsp.dsp.drc.drc.clipper
    :noindex:

    .. automethod:: process
        :noindex:

.. _lim_peak:

------------
Limiter Peak
------------

.. doxygenstruct:: limiter_t
    :members:

Will use the :ref:`env_det_peak` as an envelope to compare with the `threshold` level. According to that
will calculate the gain to apply to the sample.

.. doxygenfunction:: adsp_limiter_peak

.. autoclass:: audio_dsp.dsp.drc.drc.limiter_peak
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

-----------------
Hard Limiter Peak
-----------------

Will run :ref:`lim_peak` and clip the result if it's above the threshold.

.. doxygenfunction:: adsp_hard_limiter_peak

.. autoclass:: audio_dsp.dsp.drc.drc.hard_limiter_peak
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

-----------
Limiter RMS
-----------

Will use the :ref:`env_det_rms` as an envelope to compare with the `threshold` level. According to that
will calculate the gain to apply to the sample.

.. doxygenfunction:: adsp_limiter_rms

.. autoclass:: audio_dsp.dsp.drc.drc.limiter_rms
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

--------------
RMS Compressor
--------------

.. doxygenstruct:: compressor_t
    :members:

Will use the :ref:`env_det_rms` as an envelope to compare with the `threshold` level. According to that
will calculate the gain to apply to the sample.

.. doxygenfunction:: adsp_compressor_rms

.. autoclass:: audio_dsp.dsp.drc.drc.compressor_rms
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

------------------------
Sidechain RMS Compressor
------------------------

Takes two signals: *detect* and *input*. Will use the *detect* signal to run the :ref:`env_det_rms`,
calculate the gain and apply in to the *input* sample.

.. doxygenfunction:: adsp_compressor_rms_sidechain

.. autoclass:: audio_dsp.dsp.drc.sidechain.compressor_rms_sidechain_mono
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

=========
Expanders
=========

Exanders attenuate the signal that's below the threshold.

----------
Noise Gate
----------

Will use the :ref:`env_det_peak` as an envelope to compare with the `threshold` level. According to that
will calculate the gain to apply to the sample.

.. doxygentypedef:: noise_gate_t

.. doxygenfunction:: adsp_noise_gate

.. autoclass:: audio_dsp.dsp.drc.expander.noise_gate
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex:

-------------------------
Noise Suppressor/Expander
-------------------------

Will use the :ref:`env_det_peak` as an envelope to compare with the `threshold` level. According to that
will calculate the gain to apply to the sample.

.. doxygenstruct:: noise_suppressor_expander_t

.. doxygenfunction:: adsp_noise_suppressor_expander

.. autoclass:: audio_dsp.dsp.drc.expander.noise_suppressor_expander
    :noindex:

    .. automethod:: process
        :noindex:
    
    .. automethod:: reset_state
        :noindex:
