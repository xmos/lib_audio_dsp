########################
Python module base class
########################

All the Python DSP modules are based on a common base class.
In order to keep the documentation short, all Python classes in the
previous sections only had the ``process`` method described, and control
methods where necessary.
This section provides the user with a more in-depth information of the
Python API, which may be useful when adding custom DSP modules.

Some classes overload the base class APIs where they require different
input data types or dimensions. However, they will all have the
attributes and methods described below.

The process methods can be split into 2 groups:

1) ``process`` is a 64b floating point implementation
2) ``process_xcore`` is a 32b fixed-point implementation, with the aim of 
   being bit exact with the C/assembly implementation. 
   
The ``process_xcore`` methods can be used to simulate the xcore
implementation precision and the noise floor. The Python ``process_xcore``
implementations have very similar accuracy to the xcore 
C ``adsp_*`` implementations (subject to the module and implementation).
Python simulation methods tend to be slower as Python has a limited support
for the fixed point processing. Bit exactness is not always possible
for modules that use 32b float operations, as the rounding of these can
differ between C libraries.

There are 3 layers of ``process`` functions:

1) ``process`` operates for a single sample on a single channel
2) ``process_channels`` operates on all channels for a single sample
3) ``process_frame`` operates on all samples and channels for a single frame.

A DSP module may overload one or all of these, depending if it operates
sample-wise, channel-wise or frame-wise. It is expected that
``process_frame`` will be called by higher level DSP Stages.

.. autoclass:: audio_dsp.dsp.generic.dsp_block
    :members:
    :noindex:
