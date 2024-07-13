###############
Bi-Quad Filters
###############

============
Singe Biquad
============

A second order oder biquadratic filter. Implemented in the direct form 1.

Coefficients are stored in Q30 format to benefit from the vector unit. If coefficients don't fit
in the range of ``[-2, 1.999]`` they should be right-shifted and the shift should be passed into the API,
so it will left-shift the output. The ``state`` should be initialised to ``0``.
``state`` and ``coeffs`` must be word-aligned.

.. doxygenfunction:: adsp_biquad

.. autoclass:: audio_dsp.dsp.biquad.biquad
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: update_coeffs
        :noindex:

    .. automethod:: reset_state
        :noindex:

================
Cascaded Biquads
================

Same as the single biquad implementation but processes 8 biquad filters at a time. Can be used to implement
a simple parametric equaliser or high-order Butterworth filters.

.. doxygenfunction:: adsp_cascaded_biquads_8b

.. autoclass:: audio_dsp.dsp.cascaded_biquads.cascaded_biquads_8
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex: