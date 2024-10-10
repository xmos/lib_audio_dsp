.. _dsp_stages_section:

##########
DSP Stages
##########

DSP stages are high level blocks for use in the Python DSP
pipeline tool. Each Stage has a Python and C implementation, allowing
pipelines to be rapidly prototyped in Python before being easily 
deployed to hardware in C. The audio performance of both implementations
is equivalent.

Most stages have parameters that can be changed at runtime, and the
available parameters are outlined in the documentation.

All the DSP stages can be imported into a Python file using:

.. code-block:: console

  from audio_dsp.stages import *

The following DSP stages are available for use in the Python DSP pipeline design.

.. toctree::
    :glob:
    :maxdepth: 2

    gen/*
