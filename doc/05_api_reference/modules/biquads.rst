.. _biquad_filters:

##############
Biquad Filters
##############

.. _Biquad:

=============
Single Biquad
=============

A second order biquadratic filter, which can be used to implement many common second order filters.
The filter had been implemented in the direct form 1, and uses the xcore.ai vector unit to
calculate the 5 filter taps in a single instruction.

Coefficients are stored in Q1.30 format to benefit from the vector unit, allowing for a filter 
coefficient range of ``[-2, 1.999]``. For some high gain biquads (e.g. high shelf filters), the
numerator coefficients may exceed this range. If this is the case, the numerator coefficients only
should be right-shifted until they fit within the range (the denominator coefficients cannot become
larger than 2.0 without the poles exceeding the unit circle). The shift should be passed into the API,
and the output signal from the biquad will then have a left-shift applied. This is equivalent to
reducing the overall signal level in the biquad, then returning to unity gain afterwards. 

The ``state`` should be initialised to ``0``. The ``state`` and ``coeffs`` must be word-aligned.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_biquad

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.biquad.biquad
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: update_coeffs
            :noindex:

        .. automethod:: reset_state
            :noindex:

.. _BiquadSlew:

=====================
Single Slewing Biquad
=====================

This is similar to `Biquad`_, but when the target coefficients are updated it slew the applied
coefficients towards the new values. This can be used for real time adjustable filter control.
If the left-shift of the coefficients changes, this is managed by the ``biquad_slew_t`` object:

.. doxygenstruct:: biquad_slew_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    Filtering the samples can be carried out using ``adsp_biquad``:

    .. code-block:: C

        for (int j=0; i < n_samples; i++){
            adsp_biquad_slew_coeffs(&slew_state, states, 1);
            for (int i=0; i < n_chans; i++){
              samp_out[i, j] = adsp_biquad(samp_in[i, j], slew_state.active_coeffs, states[i], slew_state.lsh);
            }
        }

    .. doxygenfunction:: adsp_biquad_slew_init

    .. doxygenfunction:: adsp_biquad_slew_coeffs

    .. doxygenfunction:: adsp_biquad_slew_update


.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.biquad.biquad_slew
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: update_coeffs
            :noindex:

        .. automethod:: reset_state
            :noindex:


.. _CascadedBiquads:

================
Cascaded Biquads
================

The cascaded biquad module is equivalent to 8 individual biquad filters connected in series. It 
can be used to implement a simple parametric equaliser or high-order Butterworth filters,
implemented as cascaded second order sections.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_cascaded_biquads_8b

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.cascaded_biquads.cascaded_biquads_8
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:
