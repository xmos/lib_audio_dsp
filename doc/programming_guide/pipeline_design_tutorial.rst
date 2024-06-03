PIPELINE DESIGN TUTORIAL
########################

This page will walk through the basics of designing an audio DSP pipeline for the xcore with the audio_dsp
python library. This does not include adding the generated code in an application and assumes that
the reader has a reference application to run their pipeline on. Such a reference design can be found in
the xmos/sw_audio_dsp github repository.

The steps in this guide should be executed in a `Jupyter Notebook <https://jupyter.org/>`_.

Making a simple tone control
============================

Design Phase
------------

A simple yet useful DSP pipeline that could be made is a bass and treble control with output limiter. In this
design the product will stream real time audio boosting or suppressing the treble and bass and then limiting
the output amplitude to protect the output device.

The DSP pipeline will perform the following processes:

.. figure:: images/bass_treble_limit.drawio.png
   :width: 100%

   The target pipeline.


The first step is to create an instance of the :py:class:`Pipeline <audio_dsp.design.pipeline.Pipeline>` class. This
is the top level class which will be used to create and tune the pipeline. On creation the number of inputs and sample
rate must be specified.

.. code-block:: python

   from audio_dsp.design.pipeline import Pipeline

   pipeline, inputs = Pipeline.begin(
       4,          # Number of pipeline inputs.
       fs=48000    # Sample rate.
   )


The pipeline object can now be used to add DSP stages. For high shelf and low shelf use :py:class:`Biquad <audio_dsp.stages.biquad.Biquad>` and for
the limiter use :py:class:`LimiterPeak <audio_dsp.stages.limiter.LimiterPeak>`.

.. code-block:: python

    from audio_dsp.design.pipeline import Pipeline
    from audio_dsp.stages.import *

    p, i = Pipeline.begin(4, fs=48000)

    # i is a list of pipeline inputs. "lowshelf" is a label for this instance of Biquad.
    i = p.stage(Biquad, i, "lowshelf")

    # The output of lowshelf "i" is pass as the input to the
    # highshelf.
    i = p.stage(Biquad, i, "highshelf")

    # Connect highshelf to the limiter. Labels are optional
    i = p.stage(LimiterPeak, i)

    # Finally connect the last stage to the output of the pipeline.
    p.set_outputs(i)

    p.draw()


When running the above snippet in a Jupyter Notebook it will output the following image which illustrates the pipeline which has been designed:

.. figure:: images/tutorial_pipeline.png
   :width: 100%

   Generated pipeline diagram


Tuning Phase
------------

Each stage contains a number of designer methods which can be identified as they have the ``make_`` prefix. These can be used to configure
the stages. The stages also provide a ``plot_frequency_response()`` method which shows the magnitude and phase response of the stage with
its current configuration. The two biquads created above will have a flat frequency response until they are tuned. The code below shows
how to use the designer methods to convert them into the low shelf and high shelf that is desired. The individual stages are accessed using
the labels that where assigned to them when the stage was added to the pipeline.

.. code-block:: python

   # Make a low shelf with a centre frequency of 200 Hz, q of 0.7 and gain of +6 dB
   p["lowshelf"].make_lowshelf(200, 0.7, 6)
   p["lowshelf"].plot_frequency_response()

   # Make a high shelf with a centre frequency of 4000 Hz, q of 0.7 and gain of +6 dB
   p["highshelf"].make_highshelf(4000, 0.7, 6)
   p["highshelf"].plot_frequency_response()


.. figure:: images/frequency_response.png
   :width: 100%

   Frequency response of the biquads (low shelf left, high shelf right).


For this tutorial the default settings for the limiter will provide adequate performance.


Code Generation
---------------

With an initial pipeline complete, it is time to generate the xcore source code and run it on a device. The code can be generated
using the :py:class:`generate_dsp_main() <audio_dsp.design.pipeline.generate_dsp_main>` function::

    from audio_dsp.design.pipeline import generate_dsp_main
    generate_dsp_main(p)


The reference application should then provide instructions for compiling the application and running it on the target device.

With that the tuned DSP pipeline will be running on the xcore device and can be used to stream audio. The next step is to iterate on the design
and tune it to perfection. One option is to repeat the steps described above, regenerating the code with new tuning values until the performance requirements are satisfied.
But a faster option is described below which allows run time tuning of the stages in the pipeline.

Run time configuration and profiling
------------------------------------

The audio_dsp Python library provides support for interfacing with the host control application that is available with the sw_audio_dsp reference
application. There are two operations which can be performed. The first is to send new configuration to a device which is already running. As long
as the structure of the pipeline has not changed, the configuration of the pipeline can be changed in real time for convenient tuning::

    from audio_dsp.design.host_app import set_host_app
    from audio_dsp.design.pipeline import send_config_to_device, profile_pipeline

    set_host_app("path/to/dsp_host")  # pass the correct path to a host app here

This will use the host application to send the configuration to the device whilst it is running. This will not update the generated code and therefore the
device configuration will be lost when it is switched off. Rerun ``generate_dsp_main()`` in order to create an application with updated tuning parameters
baked in::

    # send the current config to the device
    send_config_to_device(p)


The second is for profiling the thread utilisation. This will display a table which reports the percentage utilisation of each thread. This number is measured
whilst the device is running, and the value displayed is the worst case that has been observed since the device booted for each thread::

    # Read back the thread utilisation
    profile_pipeline(p)

    +--------------+----------------------------------+--------------------+------------+
    | thread index | available time (ref timer ticks) | max ticks consumed | % consumed |
    +--------------+----------------------------------+--------------------+------------+
    |      0       |             2083.33              |        485         |   23.28    |
    |      1       |             2083.33              |        236         |   11.33    |
    +--------------+----------------------------------+--------------------+------------+


Designing Complex Pipelines
===========================

The audio dsp library is not limited to the simple linear pipelines shown above. Stages can scale to take an arbitrary number of inputs, and the outputs of
each stage can be split and joined arbitrarily.

Every stage has an :py:attr:`o <audio_dsp.design.stages.Stage.o>` attribute. This is an instance of :py:class:`StageOutputList <audio_dsp.design.stage.StageOutputList>`, a
container of :py:class:`StageOutput <audio_dsp.design.stage.StageOutput>`. The stage's outputs can be selected from the StageOutputList by indexing into
it, creating a new StageOutputList, which can be concatenated with other StageOutputList instances using the ``+`` operator.
When creating a stage, it will require a StageOutputList as its inputs.

.. code-block:: python

   # split the pipeline inputs
   i0 = p.stage(Biquad, i[0:2])      # use the first 2 inputs
   i1 = p.stage(Biquad, i[2])        # use the third input (index 2)
   i2 = p.stage(Biquad, i[3, 5, 6])  # use the inputs at index 3, 5, and 6
   # join biquad outputs
   i3 = p.stage(Biquad, i0 + i1 + i2[0]) # pass all of i0 and i1, as well as the first channel in i2

    p.set_outputs(i3 + i2[1:]) # The pipeline output will be all i3 channels and the 2nd and 3rd channel from i2.

As the pipeline grows it may end up consuming more MIPS than are available on a single xcore thread. The pipeline design interface allows adding additional threads
using the :py:meth:`next_thread() <audio_dsp.design.pipeline.Pipeline.next_thread>`. To determine when a new thread is used, the output of ``profile_pipeline()`` should
be observed as the pipeline grows. If a thread nears 100% utilisation then it is time to add a new thread. Each thread in the pipeline represents an xcore
hardware thread. Do not add more threads than are available in your application. The maximum number of threads that should be used, if available, is five. This
due to the architecture of the xcore processor.

.. code-block:: python

    # thread 0
    i = p.stage(Biquad, i)

    # thread 1
    p.next_thread()
    i = p.stage(Biquad, i)

    # thread 2
    p.next_thread()
    i = p.stage(Biquad, i)
