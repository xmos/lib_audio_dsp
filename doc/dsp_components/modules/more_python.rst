###################
More on Python docs
###################

In order to keep the documentation short, all python classes in the previous sections only had the ``process`` method,
and some control methods if necessary. This section is meant to provide the user with a more in-depth information of the python API.

All Python module classes are derived from the same base class. Some classes may overwrite these APIs as they would expect
different input data types/dimentions, nevertheless they will all have the following attributes and methods.

The process methods can be split into 2 groups: pure python float implementation and xcore-like implementation. The xcore-like
implementations will try to mimic the C/assembly code used to implement the module. The ``process_xcore`` methods can be used
to immulate the xcore implementation precision and the noise floor. All the ``process_xcore`` implementations tend to have a
very similar accuracy as an actual xcore implementations (subject to the module and implementation). Python emulation 
methods tend to be slower as python has a very limited support for the fixed point processing.

.. autoclass:: audio_dsp.dsp.generic.dsp_block
    :members:
    :noindex:
