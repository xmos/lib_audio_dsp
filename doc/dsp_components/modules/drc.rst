#####################
Dynamic Range Control
#####################

DSP modules that modify the signal based on the level have been classed
as dynamic range control (DRC) modules. This includes compressors, limiters
and clippers, as well as the envelope detectors used to detect the signal
level.

========================
Attack and Release Times
========================

Nearly all DRC modules feature an attack and release time to control the
responsiveness of the module to changes in signal level. Attack and
release times converted from seconds to alpha coefficients for use in
the exponential moving average. The shorter the attack or release time,
the bigger the alpha. Large alpha will result in the envelope be more
reactive to the input samples. Small alpha values will give more smoothed
behaviour. The difference between the input level and the current
envelope or gain determines whether the attack or release alpha is used.

==================
Envelope Detectors
==================

Envelope detectors run an exponential moving average (EMA) of the 
incoming signal. They are used as a part of the most DRC components.
They can also be used to implement the VU meters and level detectors.

They feature `attack and release times`_ to control the responsiveness
of the envelope detector.

The C struct below is used for all the envelope detector implementations.

.. doxygenstruct:: env_detector_t
    :members:

.. _EnvelopeDetectorPeak:

----------------------
Peak Envelope Detector
----------------------

A peak-based envelope detector will run its EMA using the absolute value of the input sample.

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
RMS Envelope Detector
---------------------

An RMS-based envelope detector will run its EMA using the square of the
input sample. It returns the mean² in order to avoid a square root.

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

A clipper limits the signal to a specified threshold. It is applied
instantaneously, so has no attack or release times.

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

Limiters will reduce the amplitude of a signal when the signal envelope
exceeds the desired threshold. This is similar behaviour to a compressor
with an infinite ratio.

A limiter will run an internal envelope detector to get the signal
envelope, then compare it to the threshold. If the envelope exceeds the
threshold, the applied gain will be reduced. If the envelope is below
the threshold, unity gain will be applied. The gain is run through an EMA
to avoid abrupt changes. The same `attack and release times`_ are used
for the envelope detector and the gain smoothing.

The C struct below is used for all the limiter implementations.

.. doxygenstruct:: limiter_t
    :members:

.. _LimiterPeak:

------------
Peak Limiter
------------

A peak limiter uses the :ref:`EnvelopeDetectorPeak` to get an envelope.
When envelope is above the threshold, the new gain is calculated as 
``threshold / envelope``.

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
Hard Peak Limiter
-----------------

A hard peak limiter is similar to a :ref:`LimiterPeak`, but will clip
the output if it's still above the threshold after the peak limiter.
This can be useful for a final output limiter before truncating any
headroom bits.

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
RMS Limiter
-----------

A RMS limiter uses the :ref:`EnvelopeDetectorRMS` to calculate an envelope.
When envelope is above the threshold, the new gain is calculated as 
``sqrt(threshold / envelope)``.

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
The C struct below is used for all the compressors implementations.

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

    .. doxygenfunction:: adsp_noise_gate

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

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
