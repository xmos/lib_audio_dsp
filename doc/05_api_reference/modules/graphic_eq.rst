
.. _geq:

==================
Graphic Equalisers
==================

.. _GraphicEq10b:


=========================
10 Band Graphic Equaliser
=========================

The graphic EQ module creates a 10 band equaliser, with octave spaced
center frequencies. This can be used to 
The equaliser is implemented as a set of parallel 4th order bandpass
filters, with a gain controlling the level of each parallel branch.
The center frequencies are:
[32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000].

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_graphic_eq_10b_init

    .. doxygenfunction:: adsp_graphic_eq_10b

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.graphic_eq.graphic_eq_10_band
        :noindex:

        .. automethod:: process
            :noindex:

        .. autoproperty:: gains_db
            :noindex:
