.. _dsp_components_section:

DSP Components
##############

`lib_audio_dsp` provides many common signal processing functions optimised
for xcore. These can be combined together to make complex audio pipelines
for many different applications, such as home audio, music production,
voice processing, and AI feature extraction.

The library is split into 2 levels of API: DSP stages and DSP modules.
Both APIs provide similar DSP functionality, but are suited to different
use cases.

The higher-level APIs are called :ref:`dsp_stages_section`. These stages are designed
to work with the Python DSP pipeline tool. This tool allows developers
to quickly and easily create, test, and deploy DSP pipelines without
needing to write a lot of code. By using DSP stages, the user can build
complex audio processing workflows in a short amount of time, making it
ideal for rapid prototyping and development.

The lower-level APIs are called :ref:`dsp_modules_section`. They are meant to be
used as an API directly in cases where the Python DSP pipeline tool is 
not used. These modules can be useful when integrating DSP functionality into an
existing system, or as a starting point for creating bespoke DSP
functions.

.. toctree::
    :maxdepth: 1

    gen/stages
    modules
    q_format
    precision
    latency
