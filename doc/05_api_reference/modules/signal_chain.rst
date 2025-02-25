.. _signal_chain:

#######################
Signal Chain Components
#######################

Signal chain components includes DSP modules for:
* combining signals, such as subtracting, adding, and mixing
* forks for splitting signals
* basic gain components, such as fixed gain, volume control, and mute
* basic delay buffers.

.. _Adder:

=====
Adder
=====

The adder will add samples from N inputs together. 
It will round and saturate the result to the Q0.31 range.

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

The subtractor will subtract one sample from another, then round and saturate the difference to Q0.31 range.

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

This module applies a fixed gain to a sample, with rounding and saturation to Q0.31 range.
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

The mixer applies a gain to all N channels of input samples and adds them together.
The sum is rounded and saturated to Q0.31 range. The gain must be in ``Q_GAIN`` format.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_mixer

    An alternative way to implement a mixer is to multiply-accumulate the
    input samples into a 64-bit word, then saturate it to a 32-bit word using:

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

The volume control allows safe real-time gain adjustments with minimal
artifacts.
When the target gain is changed, a slew is used to move from the
current gain to the target gain.
This allows smooth gain change and no clicks in the output signal.

The mute API allows the user to safely mute the signal by setting the
target gain to ``0``, with the slew ensuring no pops or clicks.
Unmuting will restore the pre-mute target gain.
The new gain can be set while muted, but will not take effect until
unmute is called.
There are separate APIs for process, setting the gain, muting and
unmuting so that volume control can easily be implemented into the
control system.

The slew is applied as an exponential of the difference between the
target and current gain.
For run-time efficiency, instead of an EMA-style alpha, the difference
is right shifted by the ``slew_shift`` parameter. The relation between
``slew_shift`` and time is further discussed in the Python class
documentation.

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

The delay module uses a memory buffer to return a sample after a specified
time period.
The returned samples will be delayed by a specified value.
The ``max_delay`` is set at initialisation, and sets the amount of
memory used by the buffers. It cannot be changed at runtime.
The current ``delay`` value can be changed at runtime within the range
``[0, max_delay]``

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


================
Switch with slew
================

The slewing switch module uses a cosine crossfade when moving switch
position in order to avoid clicks.

.. doxygenstruct:: switch_slew_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_switch_slew

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.switch_slew
        :noindex:

        .. automethod:: process_channels
            :noindex:

        .. automethod:: move_switch
            :noindex:


==========
Crossfader
==========

The crossfader mixes between two sets of inputs.

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_crossfader

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.signal_chain.crossfader
        :noindex:

        .. automethod:: process_channels
            :noindex:
