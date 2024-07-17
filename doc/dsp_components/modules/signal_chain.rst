#######################
Signal Chain Components
#######################

Signal chain components are meant to help with manage signal between the bigger stages (i.e. combine, gain, delay).

.. _Adder:

=====
Adder
=====

Will add N number of samples, round and saturate the result to the Q31 range.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_adder

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.adder
        :noindex:

        .. automethod:: process_channels
            :noindex:

.. _Subtractor:

==========
Subtractor
==========

Will subtract one sample from another, round and saturate to Q31 range.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_subtractor

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.subtractor
        :noindex:

        .. automethod:: process_channels
            :noindex:

.. _FixedGain:

==========
Fixed Gain
==========

Will apply a fixed gain to a sample, round and saturate to Q31 range.
The gain must be in ``Q_GAIN`` format.

.. doxygendefine:: Q_GAIN

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_fixed_gain

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.fixed_gain
        :noindex:

        .. automethod:: process
            :noindex:

.. _Mixer:

=====
Mixer
=====

Will apply a gain to all N samples and add then together.
Will round and saturate the output to the Q31 range. The gain must be in ``Q_GAIN`` format.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_mixer

    Another way to implement mixer will be to multiply-accumulate samples into a 64-bit word
    and saturate it to a 32-bit word using:

    .. doxygenfunction:: adsp_saturate_32b

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.mixer
        :noindex:

        .. automethod:: process_channels
            :noindex:

.. _VolumeControl:

==============
Volume Control
==============

Volume control allows save volume adjustments with minimal artefacts. It does that by keeping current and target gains.
When user sets the new gain, the target gain gets updated. Every time the new sample is being processed the volume control
will run an EMA with the current and the target gain to update the current gain. This allows smooth gain change and no clicks
in the output signal. 

Mute API allows the user to safely mute the signal by setting the target gain to ``0``. Unmute will restore the pre-mute
target gain. The new gain can be set while muted, it will take effect after the unmute is called. There are separate APIs
for process, setting the gain, muting and unmiting so that volume control can easily be implemented into the control system.

For run-time efficiency, instead of EMA alpha, this implementation uses a ``slew_shift`` parameter. The relation between
``slew_shift`` and time is further discussed in the python class documentation.

.. doxygenstruct:: volume_control_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_volume_control

    .. doxygenfunction:: adsp_volume_control_set_gain

    .. doxygenfunction:: adsp_volume_control_mute

    .. doxygenfunction:: adsp_volume_control_unmute

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

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

.. _Delay:

=====
Delay
=====

Generic delay line. The samples will be delay by a specified value. Supports ``max_delay`` and ``delay``, so that the
user can initialise an array with ``max_delay`` samples and then change the delay within the ``[0, max_delay]``
range safely.

.. doxygenstruct:: delay_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_delay

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.delay
        :noindex:

        .. automethod:: process_channels
            :noindex:

        .. automethod:: reset_state
            :noindex:

        .. automethod:: set_delay
            :noindex:
