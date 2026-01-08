# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for room reverb effects."""

from audio_dsp import _deprecated
import audio_dsp.dsp.generic as dspg
import numpy as np
import warnings
import audio_dsp.dsp.utils as utils
import audio_dsp.dsp.signal_chain as sc
from copy import deepcopy
import audio_dsp.dsp.reverb_base as rvb
import audio_dsp.dsp.filters as fltr


class allpass_fv(fltr.allpass):
    """A freeverb style all-pass filter, for use in the reverb_room
    block.

    This is only designed to be called inside reverb_room algorithms.

    """

    def __init__(self, max_delay, feedback_gain):
        # max delay cannot be changed, or you'll overflow the buffer
        self._max_delay = max_delay
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay

        self.delay = max_delay
        self.feedback = feedback_gain
        self._buffer_idx = 0

    def process(self, sample):  # type: ignore : overloads base class
        """
        Apply an all pass filter to a signal, using floating point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 1.0.

        """
        buff_out = self._buffer[self._buffer_idx]

        output = -sample + buff_out
        self._buffer[self._buffer_idx] = sample + (buff_out * self.feedback)

        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output

    def process_xcore(self, sample_int):  # type: ignore : overloads base class
        """
        Apply an all pass filter to a signal, using fixed point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 2**Q_SIG.

        Parameters
        ----------
        sample_int : int
            Input sample as an integer.

        """
        assert isinstance(sample_int, int), "Input sample must be an integer"

        buff_out = self._buffer_int[self._buffer_idx]

        # reverb pregain should be scaled so this doesn't overflow, but
        # catch it if it does
        output = utils.int64(-sample_int + buff_out)
        output = utils.saturate_int64_to_int32(output)

        # do buffer calculation in int64 accumulator so we only quantize once
        new_buff = utils.int64((sample_int << rvb.Q_VERB) + buff_out * self.feedback_int)
        self._buffer_int[self._buffer_idx] = rvb.scale_sat_int64_to_int32_floor(new_buff)

        # move buffer head
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output


class comb_fv(dspg.dsp_block):
    """A freeverb style comb filter for use in the reverb_room block.

    This is only designed to be called inside reverb_room algorithms.

    Parameters
    ----------
    max_delay : int
        Maximum delay of the comb filter.
    feedback_gain : float
        Gain applied to the delayed feedback path in the comb filter, sets
        the reverb time.
    damping : float
        Sets the low pass feedback coefficient.
    """

    def __init__(self, max_delay, feedback_gain, damping):
        # max delay cannot be changed, or you'll overflow the buffer
        self._max_delay = max_delay
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay

        self.delay = 0
        self.set_feedback(feedback_gain)

        self._buffer_idx = 0
        self._filterstore = 0.0
        self._filterstore_int = 0
        self.set_damping(damping)

    def set_delay(self, delay):
        """Set the length of the delay line. Will saturate to max_delay."""
        if delay <= self._max_delay:
            self.delay = delay
        else:
            self.delay = self._max_delay
            warnings.warn(
                "Delay cannot be greater than max delay, setting to max delay", UserWarning
            )

    def set_feedback(self, feedback):
        """Set the feedback of the comb filter, which controls the
        reverberation time.
        """
        self.feedback = feedback
        self.feedback_int = rvb.float_to_q_verb(self.feedback)

    def set_damping(self, damping):
        """Set the damping of the reverb, which controls how much high
        frequency damping is in the room. Higher damping will give
        shorter reverberation times at high frequencies.
        """
        self.damp1 = damping
        self.damp2 = 1 - self.damp1
        # super critical these add up, but also don't overflow int32...
        self.damp1_int = max(utils.int32(self.damp1 * 2**rvb.Q_VERB - 1), 1)
        self.damp2_int = utils.int32((2**31 - 1) - self.damp1_int + 1)

    def reset_state(self):
        """Reset all the delay line and filterstore values to zero."""
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay
        self._filterstore = 0.0
        self._filterstore_int = 0

    def process(self, sample):  # type: ignore : overloads base class
        """
        Apply a comb filter to a signal, using floating point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 1.0.

        """
        output = self._buffer[self._buffer_idx]

        self._filterstore = (output * self.damp2) + (self._filterstore * self.damp1)

        self._buffer[self._buffer_idx] = sample + (self._filterstore * self.feedback)

        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output

    def process_xcore(self, sample_int):  # type: ignore : overloads base class
        """
        Apply a comb filter to a signal, using fixed point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 2**Q_SIG.

        Parameters
        ----------
        sample_int : int
            Input sample as an integer.

        """
        assert isinstance(sample_int, int), "Input sample must be an integer"

        output = self._buffer_int[self._buffer_idx]

        # do state calculation in int64 accumulator so we only quantize once
        filtstore_64 = utils.int64(
            output * self.damp2_int + self._filterstore_int * self.damp1_int
        )
        self._filterstore_int = rvb.scale_sat_int64_to_int32_floor(filtstore_64)

        # do buffer calculation in int64 accumulator so we only quantize once
        new_buff = utils.int64(
            (sample_int << rvb.Q_VERB) + self._filterstore_int * self.feedback_int
        )
        self._buffer_int[self._buffer_idx] = rvb.scale_sat_int64_to_int32_floor(new_buff)

        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output


class reverb_room(rvb.reverb_base):
    """Generate a room reverb effect. This is based on Freeverb by
    Jezar at Dreampoint, and consists of 8 parallel comb filters fed
    into 4 series all-pass filters.

    Parameters
    ----------
    max_room_size : float, optional
        sets the maximum size of the delay buffers, can only be set
        at initialisation.
    room_size : float, optional
        how big the room is as a proportion of max_room_size. This
        sets delay line lengths and must be between 0 and 1.
    decay : int, optional
        The length of the reverberation of the room, between 0 and 1.
    damping : float, optional
        how much high frequency attenuation in the room, between 0 and 1

    Attributes
    ----------
    comb_lengths : np.ndarray
        An array of the comb filter delay line lengths, scaled by
        max_room_size.
    ap_length : np.ndarray
        An array of the all pass filter delay line lengths, scaled by
        max_room_size.
    combs : list
        A list of comb_fv objects containing the comb filters for the
        reverb.
    allpasses : list
        A list of allpass_fv objects containing the all pass filters for
        the reverb.
    room_size : float
    decay : float
    feedback : float
    feedback_int : int
        feedback as a fixed point integer.
    damping : float
    damping_int : int
        damping as a fixed point integer.
    """

    def __init__(
        self,
        fs,
        n_chans,
        max_room_size=1,
        room_size=1,
        decay=0.5,
        damping=0.4,
        wet_gain_db=-1,
        dry_gain_db=-1,
        pregain=0.015,
        predelay=10,
        max_predelay=None,
        Q_sig=dspg.Q_SIG,
    ):
        assert n_chans == 1, f"Reverb room only supports 1 channel. {n_chans} specified"

        # initalise wet/dry gains and predelay
        super().__init__(
            fs, n_chans, wet_gain_db, dry_gain_db, pregain, predelay, max_predelay, Q_sig
        )

        self._effect_gain = sc.fixed_gain(fs, 1, 10)

        # the magic freeverb delay line lengths are for 44.1kHz, so
        # scale them with sample rate and room size
        default_comb_lengths = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])
        default_ap_lengths = np.array([556, 441, 341, 225])

        # buffer lengths
        length_scaling = self.fs / 44100 * max_room_size
        self.comb_lengths = (default_comb_lengths * length_scaling).astype(int)
        self.ap_lengths = (default_ap_lengths * length_scaling).astype(int)

        # feedbacks
        init_fb = 0.5
        init_damping = 0.4
        self.combs = [
            comb_fv(self.comb_lengths[0], init_fb, init_damping),
            comb_fv(self.comb_lengths[1], init_fb, init_damping),
            comb_fv(self.comb_lengths[2], init_fb, init_damping),
            comb_fv(self.comb_lengths[3], init_fb, init_damping),
            comb_fv(self.comb_lengths[4], init_fb, init_damping),
            comb_fv(self.comb_lengths[5], init_fb, init_damping),
            comb_fv(self.comb_lengths[6], init_fb, init_damping),
            comb_fv(self.comb_lengths[7], init_fb, init_damping),
        ]

        feedback_ap = 0.5
        self.allpasses = [
            allpass_fv(self.ap_lengths[0], feedback_ap),
            allpass_fv(self.ap_lengths[1], feedback_ap),
            allpass_fv(self.ap_lengths[2], feedback_ap),
            allpass_fv(self.ap_lengths[3], feedback_ap),
        ]

        # set filter delays
        self.decay = decay
        self.damping = damping
        self.room_size = room_size

    def reset_state(self):
        """Reset all the delay line values to zero."""
        for cb in self.combs:
            cb.reset_state()
        for ap in self.allpasses:
            ap.reset_state()
        self._predelay.reset_state()

    def get_buffer_lens(self):
        """Get the total length of all the buffers used in the reverb."""
        total_buffers = 0
        for cb in self.combs:
            total_buffers += cb._max_delay
        for ap in self.allpasses:
            total_buffers += ap._max_delay
        return total_buffers

    @_deprecated(
        "1.0.0", "2.0.0", "Replace `reverb_room.set_pre_gain(x)` with `reverb_room.pregain = x`"
    )
    def set_pre_gain(self, pre_gain):
        """
        Set the pre gain.

        Parameters
        ----------
        pre_gain : float
            pre gain value, less than 1.
        """
        self.pregain = pre_gain

    @_deprecated(
        "1.0.0", "2.0.0", "Replace `reverb_room.set_wet_gain(x)` with `reverb_room.wet_db = x`"
    )
    def set_wet_gain(self, wet_gain_db):
        """
        Set the wet gain.

        Parameters
        ----------
        wet_gain_db : float
            Wet gain in dB, less than 0 dB.
        """
        self.wet_db = wet_gain_db

    @_deprecated(
        "1.0.0", "2.0.0", "Replace `reverb_room.set_dry_gain(x)` with `reverb_room.dry_db = x`"
    )
    def set_dry_gain(self, dry_gain_db):
        """
        Set the dry gain.

        Parameters
        ----------
        dry_gain_db : float
            Dry gain in dB, lees than 0 dB.
        """
        self.dry_db = dry_gain_db

    @property
    def decay(self):
        """The length of the reverberation of the room, between 0 and 1."""
        ret = (self.feedback - 0.7) / 0.28
        return ret

    @decay.setter
    def decay(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, rvb._LESS_THAN_1)
            warnings.warn(f"Decay {bad_x} saturates to {x}", UserWarning)
        self.feedback = x * 0.28 + 0.7

    @property
    def feedback(self):
        """Gain of the feedback line in the reverb filters. Set decay to update this value."""
        ret = float(self.combs[0].feedback)
        return ret

    @feedback.setter
    def feedback(self, x):
        for cb in self.combs:
            cb.set_feedback(x)
        self.feedback_int = self.combs[0].feedback_int

    @_deprecated(
        "1.0.0", "2.0.0", "Replace `reverb_room.set_decay(x)` with `reverb_room.decay = x`"
    )
    def set_decay(self, decay):
        """
        Set the decay of the reverb.

        Parameters
        ----------
        decay : float
            How long the reverberation of the room is, between 0 and 1.
        """
        self.decay = decay

    @property
    def damping(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self.combs[0].damp1

    @damping.setter
    def damping(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        for cb in self.combs:
            cb.set_damping(x)
        self.damping_int = self.combs[0].damp1_int

    @_deprecated(
        "1.0.0", "2.0.0", "Replace `reverb_room.set_damping(x)` with `reverb_room.damping = x`"
    )
    def set_damping(self, damping):
        """
        Set the damping of the reverb.

        Parameters
        ----------
        damping : float
            How much high frequency attenuation in the room, between 0 and 1.
        """
        self.damping = damping

    @property
    def room_size(self):
        """The room size as a proportion of the max_room_size."""
        return self._room_size

    @room_size.setter
    def room_size(self, x):
        if not (0 <= x <= 1):
            raise ValueError(
                "room_size must be between 0 and 1. For larger rooms, increase max_room size"
            )
        self._room_size = x

        comb_delays = (self.comb_lengths * self._room_size).astype(int)
        ap_delays = (self.ap_lengths * self._room_size).astype(int)

        for n in range(len(self.combs)):
            self.combs[n].set_delay(comb_delays[n])

        for n in range(len(self.allpasses)):
            self.allpasses[n].set_delay(ap_delays[n])

    @_deprecated(
        "1.0.0", "2.0.0", "Replace `reverb_room.set_room_size(x)` with `reverb_room.room_size = x`"
    )
    def set_room_size(self, room_size):
        """
        Set the current room size; will adjust the delay line lengths accordingly.

        Parameters
        ----------
        room_size : float
            How big the room is as a proportion of max_room_size. This
            sets delay line lengths and must be between 0 and 1.
        """
        self.room_size = room_size

    def process(self, sample, channel=0):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        delayed_input = self._predelay.process_channels([sample])[0]
        reverb_input = delayed_input * self.pregain

        output = 0
        for cb in self.combs:
            output += cb.process(reverb_input)

        for ap in self.allpasses:
            output = ap.process(output)

        output = self._effect_gain.process(output * self.wet) + sample * self.dry

        return output

    def process_xcore(self, sample, channel=0):
        """
        Add reverberation to a signal, using fixed point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.
        """
        sample_int = utils.float_to_fixed(sample, self.Q_sig)

        delayed_input = self._predelay.process_channels_xcore([sample_int])[0]
        reverb_input = rvb.apply_gain_xcore(delayed_input, self.pregain_int)

        output = 0
        for cb in self.combs:
            output += cb.process_xcore(reverb_input)
            utils.int64(output)

        output = utils.saturate_int64_to_int32(output)

        # these buffers are at risk of overflowing, but self.gain_int
        # should be scaled to prevent it for nearly all signals
        for ap in self.allpasses:
            output = ap.process_xcore(output)
            utils.int32(output)

        # need an extra bit in this add, if wet/dry mix is badly set
        # output can saturate (users fault)
        output = rvb.apply_gain_xcore(output, self.wet_int)
        output = self._effect_gain.process_xcore(output)
        output += rvb.apply_gain_xcore(sample_int, self.dry_int)
        utils.int64(output)
        output = utils.saturate_int64_to_int32(output)

        return utils.fixed_to_float(output, self.Q_sig)
