###############################
Finite Impulse Response Filters
###############################

.. _FirDirect:

==========
FIR Direct
==========

This library uses FIR ``filter_fir_s32`` implementation from ``lib_xcore_math`` to run on xcore.
More information on implementation can be found in 
``lib_xcore_math`` `documentation <https://www.xmos.com/view/lib_xcore_math-User_Guide>`_.

.. autoclass:: audio_dsp.dsp.fir.fir_direct
    :noindex:

    .. automethod:: process
        :noindex:
    
    .. automethod:: reset_state
        :noindex:

    .. automethod:: check_coeff_scaling
        :noindex:
