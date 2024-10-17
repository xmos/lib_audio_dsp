Precision
=========

.. note::
    For fixed point Q formats this document uses the format QM.N, where M is the number of bits
    before the decimal point (excluding the sign bit), and N is the number of bits after the decimal
    point. For an int32 number, M+N=31.

By default, the signal processing in the audio pipeline is carried out at 32 bit fixed point
precision in Q4.27 format. Assuming a 24 bit input signal in Q0.24 format, this gives 4 bits of internal headroom in
the audio pipeline, which is equivalent to 24 dB. The output of the audio pipeline will be clipped back to Q0.24 before
returning. For more precision, the pipeline can be configured to run with no headroom
in Q0.31 format, but this requires manual headroom management. More information on setting the Q
format can be found in the :ref:`library_q_format_section` section.

DSP algorithms are implemented either on the XS3 CPU or VPU (vector processing unit).

CPU algorithms are typically implemented as 32-bit x 32-bit operations into 64-bit results and
accumulators, before rounding back to 32-bit outputs.

The VPU allows for 8 simultaneous operations, with a small cost in precision. VPU algorithms are
typically implemented as 32-bit x 32-bit operations into 34-bit results and 40-bit accumulators,
before rounding back to 32-bit outputs.
