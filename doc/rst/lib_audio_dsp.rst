Lib Audio DSP
#############

.. raw:: latex

    \newpage


.. rubric:: Introduction

.. note::

  Some software components in this tool flow are prototypes and will be updated in Version 2 of the library.
  The underlying Digital Signal Processing (DSP) blocks are however fully functional. Future updates will
  enhance the features and flexibility of the design tool.

lib_audio_dsp is a DSP library for the XMOS xcore architecture. It facilitates the creation of
multithreaded audio DSP pipelines that efficiently utilise the xcore architecture.

The library is built around a set of DSP function blocks, referred to in the documentation as *Stages*,
which have a consistent API and can be combined to create many different designs.

A tool for easily
combining stages into a custom DSP pipeline is provided. DSP pipeline parameters can be adjusted and
tuned on the fly via a PC based tuning interface, and utilities for hardware controls are also provided.

lib_audio_dsp includes common signal processing functions optimised for the xcore, such as:

* biquads and FIR filters.
* compressors, limiters, noise gates and envelope detectors.
* adders, subtractors, gains, volume controls and mixers.
* delays and reverb.

These can be combined together to make complex audio pipelines for many
different applications, such as home audio, music production, voice
processing, and AI feature extraction.

This document covers the following topics:

#. :ref:`tool_user_guide_section`: A beginner's guide to installing and using the DSP design and generation Python library.
#. :ref:`design_guide_section`: Advanced guidance on designing and debugging generated DSP pipelines.
#. :ref:`dsp_components_section`: List of all DSP components and details on the backend implementation.
#. :ref:`run_time_control_guide_section`: Basic guide to add time control to a DSP application.
#. :ref:`api_reference_section`: References to DSP components, control and integration and high-level tool desing API.

The subsequent sections provide comprehensive insights into the functionalities and applications of lib_audio_dsp,
detailing how to leverage its features for efficient audio signal processing.

For example appliations, see the Application Notes related to lib_audio_dsp on the
`XMOS website <https://www.xmos.com/file/lib_audio_dsp#related-application-notes>`_.

.. toctree::
    :maxdepth: 2
    :hidden:

    01_tool_user_guide/index
    02_design_guide/index
    03_dsp_components/index
    04_run_time_control_guide/index
    05_api_reference/index


.. rubric:: Copyright & Disclaimer

|XMOS copyright|
|XMOS disclaimer|
|XMOS trademarks|
