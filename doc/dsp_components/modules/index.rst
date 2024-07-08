#########################
DSP Modules documentation
#########################

This section will be splitted into sub-sections from simplicity. 

.. toctree::
    :maxdepth: 1

    biquads.rst
    drc.rst
    reverb.rst
    signal_chain.rst

===================
How to use the docs
===================

The sub-sections will have number
of APIs each of which will have:

- C API for running a module
- Python class assosiated with a module

All Python module classes are derived from the same base class. For the readability purposes, methods
from the base class will not be shown in the DSP module classes documentation.

.. autoclass:: audio_dsp.dsp.generic.dsp_block
    :members:
    :noindex:
