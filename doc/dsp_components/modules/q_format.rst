################
Library Q format
################

Most modules in this library assume that the signal is in the specific Q format,
(others that don't, will return the same format as the one they were given).
This format is defined by the ``Q_SIG`` macro. The default exponent macro is respectivly ``SIG_EXP``.

.. doxygendefine:: Q_SIG

.. doxygendefine:: SIG_EXP

So, before (and after) any processing, the user want to be sure that their signal is in the right format.
They can either convert from their foramt to ``Q_SIG`` or change the ``Q_SIG`` to the desired value (although it's not recommended).

To convert between ``Q_SIG`` and Q31 in a save and optimised way use the APIs below.

.. doxygenfunction:: adsp_from_q31

.. doxygenfunction:: adsp_to_q31
