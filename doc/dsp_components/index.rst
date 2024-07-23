##############
DSP Components
##############

lib_audio_dsp provides many common signal processing function optimised
for xcore. These can be combined together to make complex audio pipelines
for many different applications, such as home audio, music production,
voice processing and AI feature extraction.

The library is split into 2 levels of API, DSP Stages and DSP Modules.
Both APIs provide similar DSP functionality, but are suited to different
use cases.

.. rubric:: DSP Stages

The higher-level APIs are called *DSP Stages*. These stages are designed
to work with the Python DSP pipeline tool. This tool allows developers
to quickly and easily create, test, and deploy DSP pipelines without
needing to write a lot of code. By using DSP Stages, the user can build
complex audio processing workflows in a short amount of time, making it
ideal for rapid prototyping and development.

.. rubric:: DSP Modules

The lower-level APIs are called *DSP Modules*. Those are meant to be
used as an API directly in cases where the Python DSP pipeline tool is 
not used. These can be useful when integrating DSP function into an
existing system, or as a starting point for creating bespoke DSP
functions.



.. toctree::
    :maxdepth: 1

    modules/index.rst
    stages/index.rst
    runtime_control/index.rst
