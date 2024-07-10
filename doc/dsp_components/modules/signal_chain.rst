#######################
Signal Chain Components
#######################

Signal chain components are meant to help with manage signal between the bigger stages (i.e. combine, gain, delay).

=====
Adder
=====

Will add N number of samples, round and saturate the result to the Q31 range.

.. doxygenfunction:: adsp_adder

.. autoclass:: audio_dsp.dsp.signal_chain.adder
    :noindex:

    .. automethod:: process_channels
        :noindex:

=========
Subratcor
=========

Will subract one sample form another, round and saturate to Q31 range.

.. doxygenfunction:: adsp_subtractor

.. autoclass:: audio_dsp.dsp.signal_chain.subtractor
    :noindex:

    .. automethod:: process_channels
        :noindex:

==========
Fixed Gain
==========

Will apply a fixed gain to a sample, round and saturate to Q31 range.
The gain must be in ``Q_GAIN`` format.

.. doxygendefine:: Q_GAIN

.. doxygenfunction:: adsp_fixed_gain

.. autoclass:: audio_dsp.dsp.signal_chain.fixed_gain
    :noindex:

    .. automethod:: process
        :noindex:

=====
Mixer
=====

Will appliy a gain to all N samples and add then together.
Will round and saturate the output to the Q31 range. The gain must be in ``Q_GAIN`` format.

.. doxygenfunction:: adsp_mixer

.. autoclass:: audio_dsp.dsp.signal_chain.mixer
    :noindex:

    .. automethod:: process_channels
        :noindex:

==============
Volume Control
==============

Something about volume control.

.. doxygenstruct:: volume_control_t
    :members:

.. doxygenfunction:: adsp_volume_control

.. autoclass:: audio_dsp.dsp.signal_chain.volume_control
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: set_gain
        :noindex:

    .. automethod:: mute
        :noindex:

    .. automethod:: unmute
        :noindex:

=====
Delay
=====

Something about delay.

.. doxygenstruct:: delay_t
    :members:

.. doxygenfunction:: adsp_delay

.. autoclass:: audio_dsp.dsp.signal_chain.delay
    :noindex:

    .. automethod:: process_channels
        :noindex:

    .. automethod:: reset_state
        :noindex:

    .. automethod:: set_delay
        :noindex:
