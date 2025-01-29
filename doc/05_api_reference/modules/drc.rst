.. _drc:

#####################
Dynamic Range Control
#####################

Dynamic Range Control (DRC) in audio digital signal processing (DSP) refers to the automatic adjustment of an audio
signal's amplitude to reduce its dynamic range - the difference between the loudest and quietest parts of the audio.
They include compressors, limiters and clippers, as well as the envelope detectors used to detect the signal level.

========================
Attack and Release Times
========================

Nearly all DRC modules feature an attack and release time to control the
responsiveness of the module to changes in signal level. Attack and
release times converted from seconds to alpha coefficients for use in
the the exponential moving average calculation. The shorter the attack or release time, the bigger the alpha. Large
alpha will result in the envelope becoming more reactive to the input samples. Small alpha values will give more
smoothed behaviour. The difference between the input level and the current envelope or gain determines whether the
attack or release alpha is used.

==================
Envelope Detectors
==================

Envelope detectors run an exponential moving average (EMA) of the 
incoming signal. They are used as a part of the most DRC components.
They can also be used to implement VU meters and level detectors.

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
is greater than the desired threshold. This is similar behaviour to a compressor
with an infinite ratio.

A limiter will run an internal envelope detector to get the signal
envelope, then compare it to the threshold. If the envelope is greater than the
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

A compressor will attenuate the signal when the envelope is greater than the
threshold. The input/output relationship above the threshold is defined
by the compressor ``ratio``.

As with a limiter, the compressor runs an internal envelope detector 
to get the signal envelope, then compares it to the threshold. If the
envelope is greater than the threshold, the gain will be proportionally reduced
by the ``ratio``, such that it is greater than the threshold by a smaller amount. 
If the envelope is below the threshold, unity gain will be applied. 
The gain is then run through an EMA to avoid abrupt changes, before being
applied. 

The ``ratio`` defines the input/output gradient in the logarithmic domain.
For example, a ratio of 2 will reduce the output gain by 0.5 dB for every 
1 dB the envelope is over the threshold. 
A ratio of 1 will apply no compression. 
To avoid converting the envelope to the logarithmic domain for the gain
calculation, the ratio is converted to the ``slope`` as 
``(1 - 1 / ratio) / 2`` . The gain can then be calculated as an
exponential in the linear domain.

The C struct below is used for all the compressors implementations.

.. doxygenstruct:: compressor_t
    :members:

.. _CompressorRMS:

--------------
RMS Compressor
--------------

The RMS compressor uses the :ref:`EnvelopeDetectorRMS` to calculate an
envelope.
When the envelope is above the threshold, the new gain is calculated as
``(threshold / envelope) ^ slope``.

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

The sidechain RMS compressor calculates the envelope of one signal and
uses it to compress another signal.
It takes two signals: *detect* and *input*. The envelope of the *detect* signal 
is calculated using an internal :ref:`EnvelopeDetectorRMS`.
The gain is calculated in the same way as a :ref:`CompressorRMS`, but the
gain is then applied to the *input* sample.
This can be used to reduce the level of the *input* signal when the
*detect* signal gets above the threshold.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_compressor_rms_sidechain

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.compressor_rms_sidechain_mono
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

=========
Expanders
=========

An expander attenuates a signal when the envelope is below the threshold.
This increases the dynamic range of the signal, and can be used to
attenuate quiet signals, such as low level noise.

Like limiters and compressors, an expander will run an internal envelope
detector to calculate the envelope and compare it to the threshold.
If the envelope is below the threshold, the applied gain will be reduced.
If the envelope is greater than the threshold, unity gain will be applied.
The gain is run through an EMA to avoid abrupt changes. 
The same `attack and release times`_ are used for the envelope detector
and the gain smoothing. In an expander, the attack time is defined as the
speed at which the gain returns to unity after the signal has been
below the threshold.

.. _NoiseGate:

----------
Noise Gate
----------

A noise gate uses the :ref:`EnvelopeDetectorPeak` to calculate the 
envelope of the input signal.
When the envelope  is below the threshold, a gain of 0 is applied to
the input signal. Otherwise, unity gain is applied.

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

A basic expander can also be used as a noise suppressor.
It uses the :ref:`EnvelopeDetectorPeak` to calculate the envelope of the
input signal.
When the envelope is below the threshold, the gain of the signal is 
reduced according to the ratio. Otherwise, unity gain is applied.

Like a compressor, the ``ratio`` defines the input/output gradient in
the logarithmic domain.
For example, a ratio of 2 will reduce the output gain by 0.5 dB for every 
1 dB the envelope is below the threshold. 
A ratio of 1 will apply no gain changes. 
To avoid converting the envelope to the logarithmic domain for the gain
calculation, the ratio is converted to the ``slope`` as 
``(1 - ratio)``. The gain can then be calculated as an
exponential in the linear domain.

For speed, some parameters such as  ``inv_threshold`` are computed at 
initialisation to simplify run-time computation.

.. doxygenstruct:: noise_suppressor_expander_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_noise_suppressor_expander

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.drc.noise_suppressor_expander
        :noindex:

        .. automethod:: process
            :noindex:
        
        .. automethod:: reset_state
            :noindex:
