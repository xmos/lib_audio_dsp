########################
Python module base class
########################

All the python DSP modules are based on a common base class.
In order to keep the documentation short, all python classes in the
previous sections only had the ``process`` method described, and control
methods where necessary.
This section provides the user with a more in-depth information of the
python API, which may be useful when adding custom DSP modules.

Some classes overload the base class APIs where they require different
input data types or dimensions. However, they will all have the
attributes and methods described below.

The process methods can be split into 2 groups:

1) ``process`` is a 64b floating point implementation
2) ``process_xcore`` is a 32b fixed-point implementation, with the aim of 
   being bit exact with the C/assembly implementation. 
   
The ``process_xcore`` methods can be used to simulate the xcore
implementation precision and the noise floor. The python ``process_xcore``
implementations have very similar accuracy to the xcore 
C ``adsp_*`` implementations (subject to the module and implementation).
Python simulation methods tend to be slower as python has a limited support
for the fixed point processing. Bit exactness is not always possible
for modules that use 32b float operations, as the rounding of these can
differ between C libraries.

.. autoclass:: audio_dsp.dsp.generic.dsp_block
    :members:
    :noindex:
