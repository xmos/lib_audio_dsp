.. _pipeline_design_api:

Pipeline Design API
###################

This page describes the C and Python APIs that will be needed when using the pipeline design utility.

When designing a pipeline first create an instance of :py:class:`Pipeline <audio_dsp.design.pipeline.Pipeline>`, add threads
to it with :py:meth:`Pipeline.add_thread() <audio_dsp.design.pipeline.Pipeline.add_thread>`. Then add DSP stages such as
:py:class:`Biquad <audio_dsp.stages.biquad.Biquad>` using :py:meth:`CompositeStage.stage() <audio_dsp.design.composite_stage.CompositeStage.stage>`. The pipeline can be visualised
in a `Jupyter Notebook`_ using :py:meth:`Pipeline.draw() <audio_dsp.design.pipeline.Pipeline.draw>` and the xcore source code for the
pipeline can be generated using :py:func:`generate_dsp_main() <audio_dsp.design.pipeline.generate_dsp_main>`.

Once the code is generated use the functions defined in `stages/adsp_pipeline.h`_ to read and write samples to the pipeline and update
configuration fields.

C Design API
************

.. include:: gen/api.stages.inc

Python Design API
*****************

.. include:: gen/audio_dsp.design.inc
