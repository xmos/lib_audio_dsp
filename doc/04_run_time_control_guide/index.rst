.. raw:: latex

    \newpage

.. _run_time_control_guide_section:

Run-Time Control User Guide
###########################

For many applications, the ability to update the DSP configuration at run time will be required. A simple example
would be a volume control where the end product will update the volume setting based on user input. This
DSP library has been designed with use cases like this in mind and the generated DSP pipeline provides an interface for
writing and reading the configuration of each stage.

This document details how to use this interface to extend a DSP application with run-time control
of the audio processing. For a complete example of an application that updates the DSP configuration
based on user input refer to application note AN02015.

.. toctree::
    :maxdepth: 1

    control_interface_walkthrough
