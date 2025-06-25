.. _library_q_format_section:

################
Library Q Format
################

.. note::
    For fixed point Q formats this document uses the format QM.N, where M is the number of bits
    before the decimal point (excluding the sign bit), and N is the number of bits after the decimal
    point. For an int32 number, M+N=31.

By default, the signal processing in the audio pipeline is carried out at 32 bit fixed point
precision in Q4.27 format. Assuming a 24 bit input signal in Q0.24 format, this gives 4 bits of
internal headroom in the audio pipeline.

Most modules in this library assume that the signal is in a specific global Q format.
This format is defined by the ``Q_SIG`` macro. An additional macro for the signal exponent,
``SIG_EXP`` is defined, where ``SIG_EXP = - Q_SIG``.

.. doxygendefine:: Q_SIG

.. doxygendefine:: SIG_EXP

To ensure optimal headroom and noise floor, the user should ensure that signals are in the correct
Q format before processing. Either the input Q format can be converted to ``Q_SIG``, or ``Q_SIG``
can be changed to the desired value. When using the DSP pipeline too, the Q format is automatically
converted. The user should ensure that 0 dBFS is aligned with the maximum int32 value.

.. note::
   Not using the DSP pipeline tool means that Q formats
   will not automatically be managed, and the user should take care to ensure they have the correct
   values for optimum performance and signal level.

For example, for more precision, the pipeline can be configured to run with no headroom
in Q0.31 format, but this would require manual headroom management (e.g. reducing the signal level
before a boost to avoid clipping).

If not using the DSP pipeline tool, the APIs below provide conversions between ``Q_SIG`` and Q0.31
in a safe and optimised method, assuming that 0 dBFS is aligned with the maximum int32 value.

.. doxygenfunction:: adsp_from_q31

.. doxygenfunction:: adsp_to_q31

To convert a 16 bit audio signal into the Q_SIG format and back, the following functions can be used:

.. code-block:: c

    // input signal in Q0.15 format
    int16_t signal = 1234;

    // Convert to Q0.31
    int32_t sig_32 = ((int32_t)signal) << 16;

    // Convert to Q_SIG
    sig_32 = adsp_from_q31(sig_32);

    // Do some processing here

    // Convert back to Q0.31
    sig_32 = adsp_to_q31(sig_32);

    // Convert back to 16 bit without dithering
    signal = (int16_t)(sig_32 >> 16);
