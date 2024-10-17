###############################
Finite Impulse Response Filters
###############################

Finite impulse response (FIR) filters allow the use of arbitrary filters
with a finite number of taps. This library does not provide FIR filter
design tools, but allows for coefficients to be imported from other design
tools, such as `SciPy/filter_design`_.

.. _FirDirect:

==========
FIR Direct
==========

The direct FIR implements the filter as a convolution in the time domain.
This library uses FIR ``filter_fir_s32`` implementation from ``lib_xcore_math`` to run on xcore.
More information on implementation can be found in `XCORE Math Library`_ documentation.

.. autoclass:: audio_dsp.dsp.fir.fir_direct
    :noindex:

    .. automethod:: process
        :noindex:
    
    .. automethod:: reset_state
        :noindex:

    .. automethod:: check_coeff_scaling
        :noindex:
