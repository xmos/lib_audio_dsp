.. _pipeline_design_api:

Pipeline Design API
###################

This page describes the C and Python APIs that will be needed when using the pipeline design utility.

When designing a pipeline first create an instance of ``Pipeline``, add threads to it with ``Pipeline.add_thread()``. Then add DSP stages such as ``Biquad`` using ``CompositeStage.stage()``. 
The pipeline can be visualised in a `Jupyter Notebook`_ using ``Pipeline.draw()`` and the xcore source code for the pipeline can be generated using ``generate_dsp_main()``.

.. contents::
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here

.. include:: gen/audio_dsp.design.inc
