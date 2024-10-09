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
