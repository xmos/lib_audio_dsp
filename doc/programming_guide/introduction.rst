INTRODUCTION
############

Lib audio DSP is a DSP library for the xcore. It also provides a DSP pipeline generation python library
for simple generation of mulithreaded audio DSP pipelines that efficiently utilise the xcore architecture.

This documentation separates the DSP library and pipeline generation into separate sections as it is
expected that some users will have custom use cases that require hand constructed DSP.

Adding lib_audio_dsp to your project
====================================

lib_audio_dsp has been designed to support xcommon cmake based projects. Therefore using the DSP part of this library
is as easy as adding "lib_audio_dsp" to your projects "APP_DEPENDENT_MODULES".

Using the pipeline generation utility will additionally require installing the python module that is found in the "python"
subdirectory of lib_audio_dsp. Having run cmake to download lib_audio_dsp into a sandbox, run the following command from the sandbox root::

    pip install -e lib_audio_dsp/python

It is important to re-run cmake after installing the python module so that the DSP design source files will be included in
the build. Read more on generating DSP pipelines in the :ref:`Pipeline Design Api section<pipeline_design_api>`.


