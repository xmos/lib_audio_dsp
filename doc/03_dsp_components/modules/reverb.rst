######
Reverb
######

.. _ReverbRoom:

===========
Reverb Room
===========

The room reverb module imitates the reflections of a room. The algorithm is a 
Schroeder style reverberation, based on `Freeverb by Jezar at Dreampoint <https://www.dsprelated.com/freebooks/pasp/Freeverb.html>`_.
It consists of the wet predelay, 8 parallel comb filters fed into 4 series all-pass filters,
with a wet and dry microphone control to set the effect level.


For more details on the algorithm, see `Physical Audio Signal Processing
<https://www.dsprelated.com/freebooks/pasp/Freeverb.html>`_ by Julius Smith.

.. doxygenstruct:: reverb_room_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_reverb_room

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.reverb.reverb_room
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

        .. automethod:: set_wet_dry_mix
            :noindex:

        .. automethod:: set_pre_gain
            :noindex:

        .. automethod:: set_wet_gain
            :noindex:

        .. automethod:: set_dry_gain
            :noindex:

        .. automethod:: set_decay
            :noindex:

        .. automethod:: set_damping
            :noindex:

        .. automethod:: set_room_size
            :noindex:


.. _ReverbRoomStereo:

==================
Reverb Room Stereo
==================

The stereo room reverb module extends the mono :ref:`ReverbRoom` by adding a second
set of comb and all-pass filters in parallel, and mixing the output of the
two networks. Varying the mix of the networks changes the stereo width of 
the effect.

For more details on the algorithm, see `Physical Audio Signal Processing
<https://www.dsprelated.com/freebooks/pasp/Freeverb.html>`_ by Julius Smith.


.. doxygenstruct:: reverb_room_st_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_reverb_room_st

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.reverb_stereo.reverb_room_stereo
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

        .. automethod:: set_wet_dry_mix
            :noindex:

        .. autoproperty:: wet_db
            :noindex:

        .. autoproperty:: dry_db
            :noindex:

        .. autoproperty:: decay
            :noindex:

        .. autoproperty:: damping
            :noindex:


.. _ReverbPlateStereo:

===================
Reverb Plate Stereo
===================

The plate reverb module imitates the reflections of a plate reverb,
which has more early reflections than the room reverb. Tha algorithm is
based on Dattorro's 1997 paper. This reverb consists of 4 allpass
filters for input diffusion, followed by a figure of 8 reverb tank of
allpasses, low-pass filters, and delays. The output is taken from
multiple taps in the delay lines to get a desirable echo density.
The left and right output can be mixed with various widths.

For more details on the algorithm, see
`Effect Design, Part 1: Reverberator and Other Filters
<https://aes2.org/publications/elibrary-page/?id=10160>`_ by Jon Dattorro.


.. doxygenstruct:: reverb_plate_t
    :members:

.. tab:: C API

    .. only:: latex

        .. rubric:: C API

    .. doxygenfunction:: adsp_reverb_plate

.. tab:: Python API

    .. only:: latex

        .. rubric:: Python API

    .. autoclass:: audio_dsp.dsp.reverb_plate.reverb_plate_stereo
        :noindex:

        .. automethod:: process
            :noindex:

        .. automethod:: reset_state
            :noindex:

        .. automethod:: set_wet_dry_mix
            :noindex:
