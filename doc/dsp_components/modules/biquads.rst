###############
Bi-Quad Filters
###############

================
Singe Biquad API
================

A second order oder biquadratic filter. Implemented in the form 1.

.. doxygenfunction:: adsp_biquad

.. autoclass:: audio_dsp.dsp.biquad.biquad
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: update_coeffs
        :noindex:

    .. automethod:: reset_state
        :noindex:

=====================
Cascadeds Biquads API
=====================

.. doxygenfunction:: adsp_cascaded_biquads_8b

.. autoclass:: audio_dsp.dsp.cascaded_biquads.cascaded_biquads_8
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
        :noindex: