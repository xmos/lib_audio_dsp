PIPELINE DESIGN TUTORIAL
########################

This page will walk through the basics of designing an audio DSP pipeline for the xcore with the audio_dsp
python library. This does not include adding the generated code in an application and assumes that
the reader has a reference application to run their pipeline on. Such a reference design can be found in
the xmos/sw_audio_dsp github repository.

The steps in this guide should be executed in a `Jupyter Notebook <https://jupyter.org/>`_.

Making a simple tone control
============================

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
