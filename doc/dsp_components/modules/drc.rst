#####################
Dynamic Range Control
#####################

This page contains modules that provide the dynamic adjustment according to the range of the signal.

==================
Envelope Detectors
==================

Envelope detectors run an exponential moving avarage (EMA) of the incoming signal. They are used as a part of
the most DRC components. Can also be used to implement the UV meters.

Attack and release times converted to alpha coefficients, so that the shorter the time the bigger the alpha.
Large alpha will result in the envelope be more reactive to the input samples. Attack or release alpha will
be chosen to run an EMA according to the difference of the input level and the current envelope.

The C struct below is used for all the envelope detector implementetions.

.. doxygenstruct:: env_detector_t
    :members:

.. _EnvelopeDetectorPeak:

----------------------
Envelope Detector Peak
----------------------

Peak-based envelope detector will run it's EMA using the absolute value of the input sample.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_env_detector_peak

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.envelope_detector_peak
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _EnvelopeDetectorRMS:

---------------------
Envelope Detector RMS
---------------------

RMS-based envelope detector will run it's EMA using the square of the input sample.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_env_detector_rms

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.envelope_detector_rms
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _Clipper:

=======
Clipper
=======

Will clip an input value if it's above the threshold.

.. doxygentypedef:: clipper_t

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_clipper

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.clipper
        :noindex:

        .. automethod:: process
            :noindex:


========
Limiters
========

Limiters will try to maintain the signal to be below or near the threshold. Acts as a compressor with an infinite ratio.

Will run an instance of an envelope detector to get an envelop and compare it to the threshold.
According to that, will calculate the gain to apply to the signal and run that gain through an EMA.
The EMA alphas are the same as in the envelope detectors used underneath.

The C struct below is used for all the limiter implementetions.

.. doxygenstruct:: limiter_t
    :members:

.. _LimiterPeak:

------------
Limiter Peak
------------

Will use the :ref:`EnvelopeDetectorPeak` to get an envelope. Will use the gain of ``threshold / envelope``
when envelope is above the threshold.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_limiter_peak

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.limiter_peak
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _HardLimiterPeak:

-----------------
Hard Limiter Peak
-----------------

Will run :ref:`LimiterPeak` and clip the result if it's still above the threshold.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_hard_limiter_peak

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.hard_limiter_peak
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _LimiterRMS:

-----------
Limiter RMS
-----------

Will use the :ref:`EnvelopeDetectorRMS` to get an envelope. Will use the gain of ``sqrt(threshold / envelope)``
when envelope is above the threshold.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_limiter_rms

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.limiter_rms
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

===========
Compressors
===========

Compressor will attenuate the signal above the threshold. The input/output relationship above the threshold
is defined by the compressor ``ratio``.

Similarly to the limiters, will run an instance of an envelope detector to get an envelop and compare it to the threshold.
According to that, will calculate the gain to apply to the signal and run that gain through an EMA.
The EMA alphas are the same as in the envelope detectors used underneath. The only difference with a limiter, is the
additional ``ratio`` parameter, which plays the role when calculating the gain.

Internally, the ratio is converted to the ``slope`` by using ``(1 - 1 / ratio) / 2`` convertion.
The C struct below is used for all the compressors implementetions.

.. doxygenstruct:: compressor_t
    :members:

.. _CompressorRMS:

--------------
RMS Compressor
--------------

Will use the :ref:`EnvelopeDetectorRMS` to get an envelope. Will use the gain of ``(threshold / envelope) ^ slope``
when envelope is above the threshold.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_compressor_rms

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.drc.compressor_rms
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _CompressorSidechain:

------------------------
Sidechain RMS Compressor
------------------------

Takes two signals: *detect* and *input*. Will use the *detect* signal to run the :ref:`EnvelopeDetectorRMS`.
Calculates the gain in the same way as :ref:`CompressorRMS`. Applies the EMAed gain to the *input* sample.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_compressor_rms_sidechain

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

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

Similarly to limiters and compressors will run an instance of an envelope detector to get an envelop and compare it to the threshold.
According to that, will calculate the gain to apply to the signal and run that gain through an EMA.
The EMA alphas are the same as in the envelope detectors used underneath. The difference with limiters and compressor is that
attack and release alphas are swapped so when we should normally attack, we release, and vice versa.

.. _NoiseGate:

----------
Noise Gate
----------

Will use the :ref:`EnvelopeDetectorPeak` to get an envelope. Will use the gain of ``0`` when the signal is below the threshold
and the gain of ``1`` when aboove.

.. doxygentypedef:: noise_gate_t

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. doxygenfunction:: adsp_noise_gate

    .. autoclass:: audio_dsp.dsp.drc.expander.noise_gate
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _NoiseSuppressorExpander:

-------------------------
Noise Suppressor/Expander
-------------------------

Will use the :ref:`EnvelopeDetectorPeak` to get an envelope. Will calculate the gain the the same way as :ref:`CompressorRMS`
but the ``slope`` is defined as ``1 - ratio`` as the envelope is not squared.

The ``inv_threshold`` is computed from ``threshold`` at init time to simplify run-time computation.

.. doxygenstruct:: noise_suppressor_expander_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_noise_suppressor_expander

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.expander.noise_suppressor_expander
        :noindex:

        .. automethod:: process
            :noindex:
        
        .. automethod:: reset_state
            :noindex:
