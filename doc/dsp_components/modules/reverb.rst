######
Reverb
######

.. _ReverbRoom:

===========
Reverb Room
===========

This is based on Freeverb by Jezar at Dreampoint, and consists of 8 parallel 
comb filters fed into 4 series all-pass filters.

.. doxygenstruct:: reverb_room_t
    :members:

.. doxygenfunction:: adsp_reverb_room

.. autoclass:: audio_dsp.dsp.reverb.reverb_room
    :noindex:

    .. automethod:: process
        :noindex:

    .. automethod:: reset_state
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
