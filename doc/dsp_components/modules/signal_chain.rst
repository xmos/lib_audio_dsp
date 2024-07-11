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

Volume control allows save volue adjustments with minimal artifacts. It does that by keeping current and target gains.
When user sets the new gain, the target gain gets updated. Every time the new sample is being processed the volume control
will run an EMA with the current and the target gain to update teh curretn gain. This allows smooth gain change and no clicks
in the output signal. 

Mute APi allow the user to safely mute the signal by setting the target gain to ``0``. Unmute will restore the pre-mute
target gain. The new gain can be set while mutted, it will take effect after the unmute is called. There are separate APIs
for process, setting the gain, muting and unmiting so that volume control can easily be implemented into the control system.

For run-time efficiency, intead of EMA alpha, this implementation uses a ``slew_shift`` paramenter. The relation between
``slew_shift`` and time is further discussed in the python class documentation.

.. doxygenstruct:: volume_control_t
    :members:

.. doxygenfunction:: adsp_volume_control

.. doxygenfunction:: adsp_volume_control_set_gain

.. doxygenfunction:: adsp_volume_control_mute

.. doxygenfunction:: adsp_volume_control_unmute

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

Generic delay line. The samples will be delay by a specified value. Supports ``max_delay`` and ``delay``, so that the
user can initialise an array with ``max_delay`` samples and then change the delay within the ``[0, max_delay]``
range safely.

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
