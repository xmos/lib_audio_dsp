.. _fir:

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


.. _FirBlockTD:

=====================
Block Time Domain FIR
=====================

The block time domain FIR implements the filter as a convolution in the time domain, but with a block size 
optimized for execution on the vector-unit of xcore.ai. The advantage with this one is it is over twice
the efficiency of the lib_xcore_math implementation. This block will generate C code for the block time domain FIR
filter.
More information on implementation can be found in
`AN02027: Efficient computation of FIR filters on the XCORE <https://www.xmos.com/application-notes/>`_.

.. note::

    The block time domain FIR filter is not currently implemented as a DSP Stage, so cannot be
    used with the DSP pipeline tool yet.

.. tab:: Autogenerator

    .. only:: latex
    
    .. rubric:: Autogenerator

    .. autofunction:: audio_dsp.dsp.td_block_fir.generate_td_fir
        :noindex:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: td_block_fir_data_init
    .. doxygenfunction:: td_block_fir_add_data
    .. doxygenfunction:: td_block_fir_compute

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

   .. autoclass:: audio_dsp.dsp.td_block_fir.fir_block_td
        :noindex:

        .. automethod:: process_frame
            :noindex:

        .. automethod:: reset_state
            :noindex:


.. _FirBlockFD:

==========================
Block Frequency Domain FIR
==========================

This implementation is a frequency-domain implementation resulting in a lower algorithmic
complexity than the time-domain versions. This will achieve the highest taps per second possible
with the xcore. The main cost to using this implementation is the memory requirements double
compared to the previous two time-domain versions. This block will generate C code for the block 
frequency domain FIR filter.
More information on implementation can be found in
`AN02027: Efficient computation of FIR filters on the XCORE <https://www.xmos.com/application-notes/>`_.

.. note::

    The block time domain FIR filter is not currently implemented as a DSP Stage, so cannot be
    used with the DSP pipeline tool yet.

.. tab:: Autogenerator

    .. only:: latex
    
    .. rubric:: Autogenerator

    .. autofunction:: audio_dsp.dsp.fd_block_fir.generate_fd_fir
        :noindex:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: fd_block_fir_data_init
    .. doxygenfunction:: fd_block_fir_add_data
    .. doxygenfunction:: fd_block_fir_compute

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

   .. autoclass:: audio_dsp.dsp.fd_block_fir.fir_block_fd
        :noindex:

        .. automethod:: process_frame
            :noindex:

        .. automethod:: reset_state
            :noindex: