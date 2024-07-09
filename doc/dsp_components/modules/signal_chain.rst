#######################
Signal Chain Components
#######################

=====
Adder
=====

.. doxygenfunction:: adsp_adder

.. autoclass:: audio_dsp.dsp.signal_chain.adder
    :noindex:

    .. automethod:: process_channels
        :noindex:

=========
Subratcor
=========

.. doxygenfunction:: adsp_subtractor

.. autoclass:: audio_dsp.dsp.signal_chain.subtractor
    :noindex:

    .. automethod:: process_channels
        :noindex:

==========
Fixed Gain
==========

.. doxygenfunction:: adsp_fixed_gain

.. autoclass:: audio_dsp.dsp.signal_chain.fixed_gain
    :noindex:

    .. automethod:: process
        :noindex:

=====
Mixer
=====

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
