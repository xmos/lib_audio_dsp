##########
DSP Stages
##########

DSP Stages are high level blocks for use in the Python DSP
pipeline tool. Each Stage has a Python and C implementation, allowing
pipelines to be rapidly prototyped in Python before being easily 
deployed to hardware in C. The audio performance of both implementations
is equivalent.

Most Stages have parameters that can be changed at runtime, and the
available parameters are outlined in the documentation.

The following DSP stages are available for use in the Python DSP pipeline design.

.. toctree::
    :glob:

    gen/*
