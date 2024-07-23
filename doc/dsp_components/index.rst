##############
DSP Components
##############

lib_audio_dsp provides many common signal processing function optimised
for xcore.
The library is split into 2 levels of API, both containing similar DSP 
functions. 
The higher-level APIs are called *DSP Stages*. These can be used
with the Python DSP pipeline tool to rapidly prototype and deploy DSP pipelines.
The lower-level APIs are called *DSP Modules*. Those are meant to be
used as an API directly in cases where the Python DSP pipeline tool is 
not used.

.. toctree::
    :maxdepth: 1

    modules/index.rst
    stages/index.rst
    runtime_control/index.rst
