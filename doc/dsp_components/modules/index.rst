###########
DSP Modules
###########

In lib_audio_dsp, DSP modules are the lower level functions and APIs.
These can be used directly without the pipeline building tool. The
documentation also includes more implementation details about the DSP
algorithms. It includes topics such as Q formats, C and Python APIs,
providing more detailed view of the DSP modules.

Each DSP module has been implemented in floating point Python, fixed
point int32 Python and fixed point int32 C, with optimisations for xcore.
Typically the Python and C fixed point implementations are bit exact,
allowing for Python prototyping of DSP pipelines.

.. toctree::
    :maxdepth: 1

    q_format.rst
    biquads.rst
    drc.rst
    fir.rst
    reverb.rst
    signal_chain.rst
    more_python.rst
