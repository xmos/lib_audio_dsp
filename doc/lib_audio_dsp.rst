#############
Lib Audio DSP
#############

lib_audio_dsp is a DSP library for the XMOS xcore architecture. It facilitates the creation of
multithreaded audio DSP pipelines that efficiently utilise the xcore architecture.

The library is built around a set of DSP function blocks, referred to in the documentation as “Stages” ,
which have a consistent API and can be combined to create many different designs. 

A tool for easily
combining stages into a custom DSP pipeline is provided. DSP pipeline parameters can be adjusted and 
tuned on the fly via a PC based tuning interface, and utilities for hardware controls are also provided.

lib_audio_dsp includes common signal processing functions optimised for the xcore, such as:

* biquads and FIR filters
* compressors, limiters, noise gates and envelope detectors
* adders, subtractors, gains, volume controls and mixers
* delays and reverb.

These can be combined together to make complex audio pipelines for many
different applications, such as home audio, music production, voice
processing, and AI feature extraction.

*****************
Table of Contents
*****************

.. toctree::
    :maxdepth: 2

    01_tool_user_guide/index
    02_design_guide/index
    03_dsp_components/index    
    04_run_time_control_guide/index

**********************
Copyright & Disclaimer
**********************

|XMOS copyright|
|XMOS disclaimer|
|XMOS trademarks|
