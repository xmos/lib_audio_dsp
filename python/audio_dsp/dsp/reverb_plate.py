import audio_dsp.dsp.reverb as rv
import audio_dsp.dsp.generic as dspg
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.utils as utils
from copy import deepcopy
import warnings


class lowpass_1ord(dspg.dsp_block):
    """A first order lowpass filter.

    Parameters
    ----------
    damping : float
        Sets the low pass feedback coefficient.
    """

    def __init__(self, damping):

        self._filterstore = 0.0
        self._filterstore_int = 0
        self.set_damping(damping)

    def set_damping(self, damping):
        """Set the damping of the reverb, which controls how much high
        frequency damping is in the room. Higher damping will give
        shorter reverberation times at high frequencies.
        """
        self.damp1 = damping
        self.damp2 = 1 - self.damp1
        # super critical these add up, but also don't overflow int32...
        self.damp1_int = max(utils.int32(self.damp1 * 2**rv.Q_VERB - 1), 1)
        self.damp2_int = utils.int32((2**31 - 1) - self.damp1_int + 1)

    def reset_state(self):
        """Reset all the filterstore values to zero."""
        self._filterstore = 0.0
        self._filterstore_int = 0

    def process(self, sample):  # type: ignore : overloads base class
        """
        Apply a low pass filter to a signal, using floating point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 1.0.

        """

        output = (sample * self.damp1) + (self._filterstore * self.damp2)

        self._filterstore = output

        return output

    def process_xcore(self, sample_int):  # type: ignore : overloads base class
        """
        Apply a low pass filter to a signal, using fixed point maths.

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
        output = utils.int64(
            sample_int * self.damp1_int + self._filterstore_int * self.damp2_int
        )
        output = scale_sat_int64_to_int32_floor(output)
        self._filterstore_int = output

        return output


class allpass_2(rv.allpass_fv):
    """A freeverb style all-pass filter, for use in the reverb_room block.

    Parameters
    ----------
    max_delay : int
        Maximum delay of the all-pass.
    feedback_gain : float
        Gain applied to the delayed feedback path in the all-pass. Sets
        the reverb time.
    """


    def process(self, sample):  # type: ignore : overloads base class
        """
        Apply an all pass filter to a signal, using floating point maths.

        Take one new sample and return the filtered sample.
        Input should be scaled with 0 dB = 1.0.

        """
        output = sample
        buff_out = self._buffer[self._buffer_idx]
        output -= buff_out * self.feedback

        self._buffer[self._buffer_idx] = output

        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        output = buff_out + sample*self.feedback

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
        raise NotImplementedError
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


class reverb_plate_stereo(dspg.dsp_block):
    """Generate a stereo plate reverb effect. The reverberator outputs are mixed
    according to the ``width`` parameter.

    Parameters
    ----------
    decay : int, optional
        The length of the reverberation of the room, between 0 and 1.
    damping : float, optional
        how much high frequency attenuation in the room, between 0 and 1
    width : float, optional
        how much stereo separation there is between the left and
        right channels. Setting width to 0 will yield a mono signal,
        whilst setting width to 1 will yield the most stereo
        separation.
    wet_gain_db : int, optional
        wet signal gain, less than 0 dB.
    dry_gain_db : int, optional
        dry signal gain, less than 0 dB.
    pregain : float, optional
        the amount of gain applied to the signal before being passed
        into the reverb, less than 1. If the reverb raises an
        OverflowWarning, this value should be reduced until it does not.
        The default value of 0.015 should be sufficient for most Q27
        signals.
    predelay : float, optional
        the delay applied to the wet channel in ms.
    max_predelay : float, optional
        the maximum predelay in ms.


    Attributes
    ----------
    pregain : float
    pregain_int : int
        The pregain applied before the reverb as a fixed point number.
    wet_db : float
    wet : float
    wet_int : int
        The linear gain applied to the wet signal as a fixed point
        number.
    dry : float
    dry_db : float
    dry_int : int
        The linear gain applied to the dry signal as a fixed point
        number.
    allpasses : list
        A list of allpass_fv objects containing the all pass filters for
        the reverb.
    decay : float
    damping : float
    damping_int : int
        damping as a fixed point integer.
    predelay : float
    width : float
    """

    def __init__(
        self,
        fs,
        n_chans,
        decay=0.4,
        damping=0.75,
        diffusion=0.50,
        bandwidth=0.4,
        input_diffusion_1=0.5,
        input_diffusion_2=0.5,
        width=1.0,
        wet_gain_db=-3,
        dry_gain_db=-3,
        pregain=0.005,
        predelay=10,
        max_predelay=None,
        Q_sig=dspg.Q_SIG,
    ):
        assert n_chans == 2, f"Stereo reverb only supports 2 channel. {n_chans} specified"

        dspg.dsp_block.__init__(self, fs, n_chans, Q_sig)

        # predelay
        max_predelay = predelay if max_predelay == None else max_predelay
        # single channel delay line, as input is shared
        self._predelay = sc.delay(fs, 1, max_predelay, predelay, "ms")

        self._width = width

        # gains
        self.pregain = pregain
        self.wet_db = wet_gain_db
        self.dry_db = dry_gain_db
        self._effect_gain = sc.fixed_gain(fs, n_chans, 10)

        # the dattoro delay line lengths are for 29761Hz, so
        # scale them with sample rate
        default_ap_lengths = np.array([142, 107, 379, 277, 2656, 1800])
        default_delay_lengths = np.array([4217, 4453, 3136, 3720])
        default_mod_ap_lengths = np.array([908, 672])

        # buffer lengths
        length_scaling = self.fs / 29761
        ap_lengths = (default_ap_lengths * length_scaling).astype(int)
        delay_lengths = (default_delay_lengths * length_scaling).astype(int)
        mod_ap_lengths = (default_mod_ap_lengths * length_scaling).astype(int)

        self._bandwidth = bandwidth
        self._damping = damping
        self._decay = decay
        self._diffusion = diffusion
        self._decay_diffusion_2 = np.clip(decay + 0.15, 0.25, 0.5)
        self._input_diffusion_1 = input_diffusion_1
        self._input_diffusion_2 = input_diffusion_2

        self.lowpasses = [
            lowpass_1ord(self.bandwidth),
            lowpass_1ord(1-self.damping),
            lowpass_1ord(1-self.damping),
            ]

        self.allpasses = [
            allpass_2(ap_lengths[0], self.input_diffusion_1),
            allpass_2(ap_lengths[1], self.input_diffusion_1),
            allpass_2(ap_lengths[2], self.input_diffusion_2),
            allpass_2(ap_lengths[3], self.input_diffusion_2),
            allpass_2(ap_lengths[4], self.decay_diffusion_2),
            allpass_2(ap_lengths[5], self.decay_diffusion_2),
        ]

        self.delays = [
            sc.delay(fs, 1, delay_lengths[0], delay_lengths[0], "samples"),
            sc.delay(fs, 1, delay_lengths[1], delay_lengths[1], "samples"),
            sc.delay(fs, 1, delay_lengths[2], delay_lengths[2], "samples"),
            sc.delay(fs, 1, delay_lengths[3], delay_lengths[3], "samples"),
        ]

        self.mod_allpasses = [
            allpass_2(mod_ap_lengths[0], -self.diffusion),
            allpass_2(mod_ap_lengths[1], -self.diffusion),
        ]

        default_taps_l = np.array([266, 2974, 1913, 1996, 1990, 187, 1066])
        default_taps_r = np.array([353, 3627, 1228, 2673, 2111, 335, 121])

        self.taps_l = (default_taps_l * length_scaling).astype(int)
        self.taps_r = (default_taps_r * length_scaling).astype(int)

        # get left output tap buffer lends [a, a, b, c, d, e, f]
        self.tap_lens_l = [
            self.delays[0]._max_delay,
            self.delays[0]._max_delay,
            self.allpasses[4]._max_delay,
            self.delays[1]._max_delay,
            self.delays[2]._max_delay,
            self.allpasses[5]._max_delay,
            self.delays[3]._max_delay]

        # get right output tap buffer lends [d, d, e, f, a, b, c]
        self.tap_lens_r = [
            self.delays[2]._max_delay,
            self.delays[2]._max_delay,
            self.allpasses[5]._max_delay,
            self.delays[3]._max_delay,
            self.delays[0]._max_delay,
            self.allpasses[4]._max_delay,
            self.delays[1]._max_delay]

        # buffer heads increment forwards, so set the tap starting positions to [len - tap - 1]
        for n in range(7):
            self.taps_l[n] = self.tap_lens_l[n] - self.taps_l[n]
            self.taps_r[n] = self.tap_lens_r[n] - self.taps_r[n]

    def reset_state(self):
        """Reset all the delay line values to zero."""
        for ap in self.allpasses:
            ap.reset_state()
        for de in self.delays:
            de.reset_state()
        self._predelay.reset_state()

    @property
    def dry_db(self):
        """The gain applied to the dry signal in dB."""
        return utils.db(self.dry)

    @dry_db.setter
    def dry_db(self, x):
        if x > 0:
            warnings.warn(f"Dry gain {x} saturates to 0 dB", UserWarning)
            x = 0

        self.dry = utils.db2gain(x)

    @property
    def dry(self):
        """The linear gain applied to the dry signal."""
        return self._dry

    @dry.setter
    def dry(self, x):
        self._dry = x
        self.dry_int = rv.float_to_q_verb(self.dry)

    @property
    def wet_db(self):
        """The gain applied to the wet signal in dB."""
        return utils.db(self.wet)

    @wet_db.setter
    def wet_db(self, x):
        if x > 0:
            warnings.warn(f"Wet gain {x} saturates to 0 dB", UserWarning)
            x = 0

        self.wet = utils.db2gain(x)

    @property
    def wet(self):
        """The linear gain applied to the wet signal."""
        return self._wet

    # override wet setter to also set wet_1 and wet_2
    @wet.setter
    def wet(self, x):
        self._wet = x
        self.wet_1 = self.wet * (self.width / 2 + 0.5)
        self.wet_2 = self.wet * ((1 - self.width) / 2)

        self.wet_1_int = rv.float_to_q_verb(self.wet_1)
        self.wet_2_int = rv.float_to_q_verb(self.wet_2)

    @property
    def decay(self):
        """The length of the reverberation of the room, between 0 and 1."""
        return self._decay

    @decay.setter
    def decay(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, _LESS_THAN_1)
            warnings.warn(f"Decay {bad_x} saturates to {x}", UserWarning)
        self._decay = x
        self.decay_diffusion_2 = x + 0.15

    @property
    def decay_diffusion_2(self):
        """The length of the reverberation of the room, between 0 and 1."""
        return self._decay_diffusion_2

    @decay_diffusion_2.setter
    def decay_diffusion_2(self, x):
        x = np.clip(x, 0.25, 0.5)
        self._decay_diffusion_2 = x
        self.allpasses[4].feedback = x
        self.allpasses[5].feedback = x

    @property
    def bandwidth(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        self._bandwidth = x
        self.lowpasses[0].set_damping(self.bandwidth)

    @property
    def damping(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self._damping

    @damping.setter
    def damping(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        self._damping = x
        self.lowpasses[1].set_damping(1-self.damping)
        self.lowpasses[2].set_damping(1-self.damping)

    @property
    def diffusion(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self._diffusion

    @diffusion.setter
    def diffusion(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        self._diffusion = x
        self.mod_allpasses[0].feedback = -self.diffusion
        self.mod_allpasses[1].feedback = -self.diffusion

    @property
    def input_diffusion_1(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self._input_diffusion_1

    @input_diffusion_1.setter
    def input_diffusion_1(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        self._input_diffusion_1 = x
        self.allpasses[0].feedback = self.input_diffusion_1
        self.allpasses[1].feedback = self.input_diffusion_1

    @property
    def input_diffusion_2(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self._input_diffusion_2

    @input_diffusion_2.setter
    def input_diffusion_2(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        self._input_diffusion_2 = x
        self.allpasses[2].feedback = self.input_diffusion_2
        self.allpasses[3].feedback = self.input_diffusion_2

    @property
    def wet(self):
        """The linear gain applied to the wet signal."""
        return self._wet

    # override wet setter to also set wet_1 and wet_2
    @wet.setter
    def wet(self, x):
        self._wet = x
        self.wet_1 = self.wet * (self.width / 2 + 0.5)
        self.wet_2 = self.wet * ((1 - self.width) / 2)

        self.wet_1_int = rv.float_to_q_verb(self.wet_1)
        self.wet_2_int = rv.float_to_q_verb(self.wet_2)

    @property
    def width(self):
        """Stereo separation of the reverberated signal."""
        return self._width

    @width.setter
    def width(self, value):
        if not (0 <= value <= 1):
            raise ValueError("width must be between 0 and 1.")
        self._width = value
        # recalculate wet gains
        self.wet = self.wet

    @property
    def pregain(self):
        """
        The pregain applied before the reverb as a floating point
        number.
        """
        return self._pregain

    @pregain.setter
    def pregain(self, x):
        if not (0 <= x < 1):
            bad_x = x
            x = np.clip(x, 0, _LESS_THAN_1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)

        self._pregain = x
        self.pregain_int = utils.int32(x * 2**rv.Q_VERB)

    def set_wet_dry_mix(self, mix):
        """
        Will mix wet and dry signal by adjusting wet and dry gains.
        So that when the mix is 0, the output signal is fully dry,
        when 1, the output signal is fully wet. Tries to maintain a
        stable signal level using -4.5 dB Pan Law.

        Parameters
        ----------
        mix : float
            The wet/dry mix, must be [0, 1].
        """
        if not (0 <= mix <= 1):
            bad_mix = mix
            mix = np.clip(mix, 0, 1)
            warnings.warn(f"Wet/dry mix {bad_mix} saturates to {mix}", UserWarning)
        # get an angle [0, pi /2]
        omega = mix * np.pi / 2

        # -4.5 dB
        self.dry = np.sqrt((1 - mix) * np.cos(omega))
        self.wet = np.sqrt(mix * np.sin(omega))
        # there's an extra gain of 10 dB added to the wet channel to
        # make it similar level to the dry, so that the mixing is smooth.
        # Couldn't add it to the wet gain itself as it's in q31

    def process(self, sample, channel=0):
        """Process is not implemented for the stereo reverb, as it needs
        2 channels at once.
        """
        raise NotImplementedError

    def process_xcore(self, sample, channel=0):
        """process_xcore is not implemented for the stereo reverb, as it needs
        2 channels at once.
        """
        raise NotImplementedError

    def process_channels(self, sample_list: list[float]):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        reverb_input = (sample_list[0] + sample_list[1]) * self.pregain
        reverb_input = self._predelay.process_channels([reverb_input])[0]

        reverb_input = self.lowpasses[0].process(reverb_input)

        for n in range(4):
            reverb_input = self.allpasses[n].process(reverb_input)

        path_1 = deepcopy(reverb_input)
        idx = self.delays[3].buffer_idx
        path_1 += self.decay * self.delays[3].buffer[0, idx]

        path_2 = deepcopy(reverb_input)
        idx = self.delays[1].buffer_idx
        path_2 += self.decay * self.delays[1].buffer[0, idx]

        path_1 = self.mod_allpasses[0].process(path_1)
        path_1 = self.delays[0].process_channels([path_1])[0]
        path_1 = self.lowpasses[1].process(path_1)
        path_1 *= self.decay
        path_1 = self.allpasses[4].process(path_1)
        path_1 = self.delays[1].process_channels([path_1])[0]

        path_2 = self.mod_allpasses[1].process(path_2)
        path_2 = self.delays[2].process_channels([path_2])[0]
        path_2 = self.lowpasses[2].process(path_2)
        path_2 *= self.decay
        path_2 = self.allpasses[5].process(path_2)
        path_2 = self.delays[3].process_channels([path_2])[0]

        # get the output taps
        output_l = 0.6*self.delays[0].buffer[0, self.taps_l[0]]
        output_l += 0.6* self.delays[0].buffer[0, self.taps_l[1]]
        output_l -= 0.6* self.allpasses[4]._buffer[self.taps_l[2]]
        output_l += 0.6* self.delays[1].buffer[0, self.taps_l[3]]
        output_l -= 0.6* self.delays[2].buffer[0, self.taps_l[4]]
        output_l -= 0.6* self.allpasses[5]._buffer[self.taps_l[5]]
        output_l -= 0.6* self.delays[3].buffer[0, self.taps_l[6]]

        output_r = 0.6*self.delays[2].buffer[0, self.taps_r[0]]
        output_r += 0.6*self.delays[2].buffer[0, self.taps_r[1]]
        output_r -= 0.6*self.allpasses[5]._buffer[self.taps_r[2]]
        output_r += 0.6*self.delays[3].buffer[0, self.taps_r[3]]
        output_r -= 0.6*self.delays[0].buffer[0, self.taps_r[4]]
        output_r -= 0.6*self.allpasses[4]._buffer[self.taps_r[5]]
        output_r -= 0.6*self.delays[1].buffer[0, self.taps_r[6]]

        # move output taps
        for n in range(7):
            self.taps_l[n] += 1
            self.taps_r[n] += 1
            if self.taps_l[n] >= self.tap_lens_l[n]:
                self.taps_l[n] = 0
            if self.taps_r[n] >= self.tap_lens_r[n]:
                self.taps_r[n] = 0

        # stereo width control
        output_l_final = output_l * self.wet_1 + output_r * self.wet_2
        output_l_final = self._effect_gain.process(output_l_final) + sample_list[0] * self.dry
        # output_l_final = output_l*self.wet + sample_list[0] * self.dry
        output_r = output_r * self.wet_1 + output_l * self.wet_2
        output_r = self._effect_gain.process(output_r) + sample_list[1] * self.dry
        # output_r = output_r*self.wet + sample_list[1] * self.dry

        return [output_l_final, output_r]

    def process_channels_xcore(self, sample_list: list[float]):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        sample_list_int = utils.float_list_to_int32(sample_list, self.Q_sig)

        acc = 1 << (rv.Q_VERB - 1)
        acc += sample_list_int[0] * self.pregain_int
        acc += sample_list_int[1] * self.pregain_int
        utils.int64(acc)
        reverb_input = utils.int32_mult_sat_extract(acc, 1, rv.Q_VERB)
        reverb_input = self._predelay.process_channels_xcore([reverb_input])[0]

        raise NotImplementedError

        output_l = 0
        output_r = 0
        for n in range(len(self.combs_l)):
            output_l += self.combs_l[n].process_xcore(reverb_input)
            output_r += self.combs_r[n].process_xcore(reverb_input)
            utils.int64(output_l)
            utils.int64(output_r)

        output_l = utils.saturate_int64_to_int32(output_l)
        output_r = utils.saturate_int64_to_int32(output_r)

        for n in range(len(self.allpasses_l)):
            output_l = self.allpasses_l[n].process_xcore(output_l)
            output_r = self.allpasses_r[n].process_xcore(output_r)
            utils.int32(output_l)
            utils.int32(output_r)

        output_l_final = _2maccs_sat_xcore(output_l, output_r, self.wet_1_int, self.wet_2_int)
        output_l_final = self._effect_gain.process_xcore(output_l_final)
        output_l_final += rv.apply_gain_xcore(sample_list_int[0], self.dry_int)
        utils.int64(output_l_final)
        output_l_final = utils.saturate_int64_to_int32(output_l_final)

        output_r = _2maccs_sat_xcore(output_r, output_l, self.wet_1_int, self.wet_2_int)
        output_r = self._effect_gain.process_xcore(output_r)
        output_r += rv.apply_gain_xcore(sample_list_int[1], self.dry_int)
        utils.int64(output_r)
        output_r = utils.saturate_int64_to_int32(output_r)

        output_l_flt = utils.int32_to_float(output_l_final, self.Q_sig)
        output_r_flt = utils.int32_to_float(output_r, self.Q_sig)

        return [output_l_flt, output_r_flt]

    def process_frame(self, frame: list[np.ndarray]):
        """
        Take a list frames of samples and return the processed frames.

        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_channels only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels([frame[0][sample], frame[1][sample]])
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]
        return output

    def process_frame_xcore(self, frame: list[np.ndarray]):
        """
        Take a list frames of samples and return the processed frames,
        using a bit exact xcore implementation.
        A frame is defined as a list of 1-D numpy arrays, where the
        number of arrays is equal to the number of channels, and the
        length of the arrays is equal to the frame size.

        When calling self.process_channel_xcore only take the first output.

        """
        n_outputs = len(frame)
        assert n_outputs == 2, "has to be stereo"
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for sample in range(frame_size):
            out_samples = self.process_channels_xcore([frame[0][sample], frame[1][sample]])
            output[0][sample] = out_samples[0]
            output[1][sample] = out_samples[1]

        return output

if __name__ == "__main__":
    import soundfile as sf
    import numpy as np
    in_wav, fs = sf.read(r"C:\Users\allanskellett\Documents\046_FRJ\sing_test (1).wav")
    rvp  = reverb_plate_stereo(fs, 2)
    in_wav = np.stack((in_wav, in_wav), 1)
    output_flt = np.zeros_like(in_wav)
    for n in range(in_wav.shape[0]):
        output_flt[n] = rvp.process_channels(in_wav[n])

    sf.write("plate.wav", output_flt, fs)