# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for reverb effects."""

from audio_dsp import _deprecated
import audio_dsp.dsp.generic as dspg
import numpy as np
import warnings
import audio_dsp.dsp.utils as utils
from copy import deepcopy
from deprecation import deprecated

Q_VERB = 31


def apply_gain_xcore(sample, gain):
    """Apply the gain to a sample using fixed-point math. Assumes that gain is in Q_VERB format."""
    acc = 1 << (Q_VERB - 1)
    acc += sample * gain
    y = utils.int32_mult_sat_extract(acc, 1, Q_VERB)
    return y


def scale_sat_int64_to_int32_floor(val):
    """Quanitze an int64 to int32, saturating and quantizing to zero
    in the process. This is useful for feedback paths, where limit
    cycles can occur if you don't round to zero.
    """
    # force the comb filter/all pass feedback to converge to zero and
    # avoid limit noise by rounding to zero. Above 0, truncation does
    # this, but below 0 we truncate to -inf, so add just under 1 to go
    # up instead.
    if val < 0:
        val += (2**Q_VERB) - 1

    # saturate
    if val > (2 ** (31 + Q_VERB) - 1):
        warnings.warn("Saturation occurred", utils.SaturationWarning)
        val = 2 ** (31 + Q_VERB) - 1
    elif val < -(2 ** (31 + Q_VERB)):
        warnings.warn("Saturation occurred", utils.SaturationWarning)
        val = -(2 ** (31 + Q_VERB))

    # shift to int32
    y = utils.int32(val >> Q_VERB)

    return y


class allpass_fv(dspg.dsp_block):
    """A freeverb style all-pass filter, for use in the reverb_room block.

    Parameters
    ----------
    max_delay : int
        Maximum delay of the all-pass.
    feedback_gain : float
        Gain applied to the delayed feedback path in the all-pass. Sets
        the reverb time.
    """

    def __init__(self, max_delay, feedback_gain):
        # max delay cannot be changed, or you'll overflow the buffer
        self._max_delay = max_delay
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay

        self.delay = 0
        self.feedback = feedback_gain
        self.feedback_int = utils.int32(self.feedback * 2**Q_VERB)
        self._buffer_idx = 0

    def set_delay(self, delay):
        """Set the length of the delay line. Will saturate to max_delay."""
        if delay < self._max_delay:
            self.delay = delay
        else:
            self.delay = self._max_delay
            Warning("Delay cannot be greater than max delay, setting to max delay")

    def reset_state(self):
        """Reset all the delay line values to zero."""
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay
        return

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
        new_buff = utils.int64((sample_int << Q_VERB) + buff_out * self.feedback_int)
        self._buffer_int[self._buffer_idx] = scale_sat_int64_to_int32_floor(new_buff)

        # move buffer head
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output


class comb_fv(dspg.dsp_block):
    """A freeverb style comb filter for use in the reverb_room block.

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
        if delay < self._max_delay:
            self.delay = delay
        else:
            self.delay = self._max_delay
            Warning("Delay cannot be greater than max delay, setting to max delay")

    def set_feedback(self, feedback):
        """Set the feedback of the comb filter, which controls the
        reverberation time.
        """
        self.feedback = feedback
        self.feedback_int = utils.int32(self.feedback * 2**Q_VERB)

    def set_damping(self, damping):
        """Set the damping of the reverb, which controls how much high
        frequency damping is in the room. Higher damping will give
        shorter reverberation times at high frequencies.
        """
        self.damp1 = damping
        self.damp2 = 1 - self.damp1
        # super critical these add up, but also don't overflow int32...
        self.damp1_int = max(utils.int32(self.damp1 * 2**Q_VERB - 1), 1)
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
        self._filterstore_int = scale_sat_int64_to_int32_floor(filtstore_64)

        # do buffer calculation in int64 accumulator so we only quantize once
        new_buff = utils.int64((sample_int << Q_VERB) + self._filterstore_int * self.feedback_int)
        self._buffer_int[self._buffer_idx] = scale_sat_int64_to_int32_floor(new_buff)

        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output


class reverb_room(dspg.dsp_block):
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
        how long the reverberation of the room is, between 0 and 1
    damping : float, optional
        how much high frequency attenuation in the room, between 0 and 1
    wet_gain_db : int, optional
        wet signal gain, less than 0 dB.
    dry_gain_db : int, optional
        dry signal gain, less than 0 dB.
    pregain : float, optional
        the amount of gain applied to the signal before being passed
        into the reverb, less than 1.


    Attributes
    ----------
    pregain : float
        The pregain applied before the reverb as a floating point
        number.
    pregain_int : int
        The pregain applied before the reverb as a fixed point number.
    wet : float
        The linear gain applied to the wet signal as a floating point
        number.
    wet_int : int
        The linear gain applied to the wet signal as a fixed point
        number.
    dry : float
        The linear gain applied to the dry signal as a floating point
        number.
    dry_int : int
        The linear gain applied to the dry signal as a fixed point
        number.
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
        The room size as a proportion of the max_room_size.
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
        Q_sig=dspg.Q_SIG,
    ):
        assert n_chans == 1, f"Reverb only supports 1 channel. {n_chans} specified"

        super().__init__(fs, 1, Q_sig)

        # gains
        self.set_pre_gain(pregain)
        self.set_wet_gain(wet_gain_db)
        self.set_dry_gain(dry_gain_db)

        # the magic freeverb delay line lengths are for 44.1kHz, so
        # scale them with sample rate and room size
        default_comb_lengths = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])
        default_ap_lengths = np.array([556, 441, 341, 225])

        # buffer lengths
        length_scaling = self.fs / 44100 * max_room_size
        self.comb_lengths = (default_comb_lengths * length_scaling).astype(int)
        self.ap_lengths = (default_ap_lengths * length_scaling).astype(int)

        # feedbacks
        if not (0 <= decay <= 1):
            raise ValueError("Decay must be between 0 and 1")
        feedback_cb = decay * 0.28 + 0.7  # avoids too much or too little feedback
        feedback_ap = 0.5

        self.combs = [
            comb_fv(self.comb_lengths[0], feedback_cb, damping),
            comb_fv(self.comb_lengths[1], feedback_cb, damping),
            comb_fv(self.comb_lengths[2], feedback_cb, damping),
            comb_fv(self.comb_lengths[3], feedback_cb, damping),
            comb_fv(self.comb_lengths[4], feedback_cb, damping),
            comb_fv(self.comb_lengths[5], feedback_cb, damping),
            comb_fv(self.comb_lengths[6], feedback_cb, damping),
            comb_fv(self.comb_lengths[7], feedback_cb, damping),
        ]

        self.allpasses = [
            allpass_fv(self.ap_lengths[0], feedback_ap),
            allpass_fv(self.ap_lengths[1], feedback_ap),
            allpass_fv(self.ap_lengths[2], feedback_ap),
            allpass_fv(self.ap_lengths[3], feedback_ap),
        ]

        # set filter delays
        self.set_room_size(room_size)

    def reset_state(self):
        """Reset all the delay line values to zero."""
        for cb in self.combs:
            cb.reset_state()
        for ap in self.allpasses:
            ap.reset_state()

    def get_buffer_lens(self):
        """Get the total length of all the buffers used in the reverb."""
        total_buffers = 0
        for cb in self.combs:
            total_buffers += cb._max_delay
        for ap in self.allpasses:
            total_buffers += ap._max_delay
        return total_buffers

    @_deprecated("1.0.0", "2.0.0", "Use the pre_gain property")
    def set_pre_gain(self, pre_gain):
        """
        Set the pre gain.

        Parameters
        ----------
        pre_gain : float
            pre gain value, less than 1.
        """
        if not (0 <= pre_gain < 1):
            raise ValueError("Pre gain must be less than 1 and positive")

        self.pregain = pre_gain
        self.pregain_int = utils.int32(self.pregain * 2**Q_VERB)

    def set_wet_gain(self, wet_gain_db):
        """
        Set the wet gain.

        Parameters
        ----------
        wet_gain_db : float
            Wet gain in dB, less than 0 dB.
        """
        if wet_gain_db > 0:
            raise ValueError("Wet gain must be less than 0 dB")

        self.wet = utils.db2gain(wet_gain_db)
        self.wet_int = utils.int32((self.wet * 2**Q_VERB) - 1)

    def set_dry_gain(self, dry_gain_db):
        """
        Set the dry gain.

        Parameters
        ----------
        dry_gain_db : float
            Dry gain in dB, lees than 0 dB.
        """
        if dry_gain_db > 0:
            raise ValueError("Dry gain must be less than 0 dB")

        self.dry = utils.db2gain(dry_gain_db)
        self.dry_int = utils.int32((self.dry * 2**Q_VERB) - 1)

    def set_decay(self, decay):
        """
        Set the decay of the reverb.

        Parameters
        ----------
        decay : float
            How long the reverberation of the room is, between 0 and 1.
        """
        if not (0 <= decay <= 1):
            raise ValueError("Decay must be between 0 and 1")
        # avoids too much or too little feedback
        feedback_cb = decay * 0.28 + 0.7
        for cb in self.combs:
            cb.set_feedback(feedback_cb)

    def set_damping(self, damping):
        """
        Set the damping of the reverb.

        Parameters
        ----------
        damping : float
            How much high frequency attenuation in the room, between 0 and 1.
        """
        if not (0 <= damping <= 1):
            raise ValueError("Damping must be between 0 and 1")
        for cb in self.combs:
            cb.set_damping(damping)

    def set_room_size(self, room_size):
        """
        Set the current room size; will adjust the delay line lengths accordingly.

        Parameters
        ----------
        room_size : float
            How big the room is as a proportion of max_room_size. This
            sets delay line lengths and must be between 0 and 1.
        """
        if not (0 <= room_size <= 1):
            raise ValueError(
                "room_size must be between 0 and 1. For larger rooms, increase max_room size"
            )
        self.room_size = room_size

        comb_delays = (self.comb_lengths * self.room_size).astype(int)
        ap_delays = (self.ap_lengths * self.room_size).astype(int)

        for n in range(len(self.combs)):
            self.combs[n].set_delay(comb_delays[n])

        for n in range(len(self.allpasses)):
            self.allpasses[n].set_delay(ap_delays[n])

    def process_frame(self, frame):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process only take the first output.

        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        if n_outputs == 1:
            for sample in range(frame_size):
                output[0][sample] = self.process(output[0][sample])
        else:
            for chan in range(n_outputs):
                this_chan = output[chan]
                for sample in range(frame_size):
                    this_chan[sample] = self.process(this_chan[sample], channel=chan)

        return output

    def process(self, sample, channel=0):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        reverb_input = sample * self.pregain

        output = 0
        for cb in self.combs:
            output += cb.process(reverb_input)

        for ap in self.allpasses:
            output = ap.process(output)

        output = output * self.wet + sample * self.dry
        return output

    def process_xcore(self, sample, channel=0):
        """
        Add reverberation to a signal, using fixed point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.
        """
        sample_int = utils.float_to_int32(sample, self.Q_sig)

        reverb_input = apply_gain_xcore(sample_int, self.pregain_int)

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
        output = apply_gain_xcore(output, self.wet_int)
        output += apply_gain_xcore(sample_int, self.dry_int)
        utils.int64(output)
        output = utils.saturate_int64_to_int32(output)

        return utils.int32_to_float(output, self.Q_sig)
