# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for plate reverb effects."""

import audio_dsp.dsp.generic as dspg
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.utils as utils
from copy import deepcopy
import warnings
import audio_dsp.dsp.reverb_base as rvb
import audio_dsp.dsp.reverb as rv
import audio_dsp.dsp.filters as fltr


class reverb_plate_stereo(rvb.reverb_stereo_base):
    """Generate a stereo plate reverb effect, based on Dattorro's 1997
    paper. This reverb consists of 4 allpass filters for input diffusion,
    followed by a figure of 8 reverb tank of allpasses, low-pass filters,
    and delays. The output is taken from multiple taps in the delay lines
    to get a good echo density.

    Parameters
    ----------
    decay : int, optional
        The length of the reverberation of the room, between 0 and 1.
    damping : float, optional
        how much high frequency attenuation in the room, between 0 and 1
    pregain : float, optional
        the amount of gain applied to the signal before being passed
        into the reverb, less than 1. If the reverb raises an
        OverflowWarning, this value should be reduced until it does not.
        The default value of 0.5 should be sufficient for most Q27
        signals, and should be reduced by 1 bit per increase in Q format,
        e.g. 0.25 for Q28, 0.125 for Q29 etc.

    Attributes
    ----------
    allpasses : list
        A list of allpass objects containing the all pass filters for
        the reverb.
    decay : float
    damping : float
    damping_int : int
        damping as a fixed point integer.
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
        pregain=0.5,
        predelay=10,
        max_predelay=None,
        Q_sig=dspg.Q_SIG,
    ):
        assert n_chans == 2, f"Stereo reverb only supports 2 channel. {n_chans} specified"

        # initalise wet/dry gains, width, and predelay
        super().__init__(
            fs, n_chans, width, wet_gain_db, dry_gain_db, pregain, predelay, max_predelay, Q_sig
        )

        self._effect_gain = sc.fixed_gain(fs, n_chans, -1)

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
        self._diffusion = diffusion
        self._decay_diffusion_2 = np.clip(decay + 0.15, 0.25, 0.5)
        self._input_diffusion_1 = input_diffusion_1
        self._input_diffusion_2 = input_diffusion_2

        self.lowpasses = [
            fltr.lowpass_1ord(fs, 1, self.bandwidth),
            fltr.lowpass_1ord(fs, 1, 1 - self.damping),
            fltr.lowpass_1ord(fs, 1, 1 - self.damping),
        ]

        self.allpasses = [
            fltr.allpass(fs, 1, ap_lengths[0], self.input_diffusion_1),
            fltr.allpass(fs, 1, ap_lengths[1], self.input_diffusion_1),
            fltr.allpass(fs, 1, ap_lengths[2], self.input_diffusion_2),
            fltr.allpass(fs, 1, ap_lengths[3], self.input_diffusion_2),
            fltr.allpass(fs, 1, ap_lengths[4], self.decay_diffusion_2),
            fltr.allpass(fs, 1, ap_lengths[5], self.decay_diffusion_2),
        ]

        self.delays = [
            sc.delay(fs, 1, delay_lengths[0], delay_lengths[0], "samples"),
            sc.delay(fs, 1, delay_lengths[1], delay_lengths[1], "samples"),
            sc.delay(fs, 1, delay_lengths[2], delay_lengths[2], "samples"),
            sc.delay(fs, 1, delay_lengths[3], delay_lengths[3], "samples"),
        ]

        self.mod_allpasses = [
            fltr.allpass(fs, 1, mod_ap_lengths[0], -self.diffusion),
            fltr.allpass(fs, 1, mod_ap_lengths[1], -self.diffusion),
        ]

        self.decay = decay

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
            self.delays[3]._max_delay,
        ]

        # get right output tap buffer lends [d, d, e, f, a, b, c]
        self.tap_lens_r = [
            self.delays[2]._max_delay,
            self.delays[2]._max_delay,
            self.allpasses[5]._max_delay,
            self.delays[3]._max_delay,
            self.delays[0]._max_delay,
            self.allpasses[4]._max_delay,
            self.delays[1]._max_delay,
        ]

        # buffer heads increment forwards, so set the tap starting positions to [len - tap - 1]
        for n in range(7):
            self.taps_l[n] = self.tap_lens_l[n] - self.taps_l[n]
            self.taps_r[n] = self.tap_lens_r[n] - self.taps_r[n]

        self.bandwidth = bandwidth
        self.damping = damping
        self.diffusion = diffusion
        self.input_diffusion_1 = input_diffusion_1
        self.input_diffusion_2 = input_diffusion_2

    def reset_state(self):
        """Reset all the delay line values to zero."""
        for ap in self.allpasses:
            ap.reset_state()
        for ap in self.mod_allpasses:
            ap.reset_state()
        for de in self.delays:
            de.reset_state()
        for lp in self.lowpasses:
            lp.reset_state()
        self._predelay.reset_state()

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
        self.decay_int = rvb.float_to_q_verb(x)
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
        self.lowpasses[0].set_bandwidth(self.bandwidth)

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
        self.lowpasses[1].set_bandwidth(1 - self.damping)
        self.lowpasses[2].set_bandwidth(1 - self.damping)

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
        output_l = 0.6 * self.delays[0].buffer[0, self.taps_l[0]]
        output_l += 0.6 * self.delays[0].buffer[0, self.taps_l[1]]
        output_l -= 0.6 * self.allpasses[4]._buffer[self.taps_l[2]]
        output_l += 0.6 * self.delays[1].buffer[0, self.taps_l[3]]
        output_l -= 0.6 * self.delays[2].buffer[0, self.taps_l[4]]
        output_l -= 0.6 * self.allpasses[5]._buffer[self.taps_l[5]]
        output_l -= 0.6 * self.delays[3].buffer[0, self.taps_l[6]]

        output_r = 0.6 * self.delays[2].buffer[0, self.taps_r[0]]
        output_r += 0.6 * self.delays[2].buffer[0, self.taps_r[1]]
        output_r -= 0.6 * self.allpasses[5]._buffer[self.taps_r[2]]
        output_r += 0.6 * self.delays[3].buffer[0, self.taps_r[3]]
        output_r -= 0.6 * self.delays[0].buffer[0, self.taps_r[4]]
        output_r -= 0.6 * self.allpasses[4]._buffer[self.taps_r[5]]
        output_r -= 0.6 * self.delays[1].buffer[0, self.taps_r[6]]

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

        acc = 1 << (rvb.Q_VERB - 1)
        acc += sample_list_int[0] * self.pregain_int
        acc += sample_list_int[1] * self.pregain_int
        utils.int64(acc)
        reverb_input = utils.int32_mult_sat_extract(acc, 1, rvb.Q_VERB)
        reverb_input = self._predelay.process_channels_xcore([reverb_input])[0]

        reverb_input = self.lowpasses[0].process_xcore(reverb_input)

        for n in range(4):
            reverb_input = self.allpasses[n].process_xcore(reverb_input)

        idx = self.delays[3].buffer_idx
        path_1 = utils.int64(
            (reverb_input << rvb.Q_VERB) + self.decay_int * self.delays[3].buffer[0, idx]
        )
        path_1 = rvb.scale_sat_int64_to_int32_floor(path_1)

        idx = self.delays[1].buffer_idx
        path_2 = utils.int64(
            (reverb_input << rvb.Q_VERB) + self.decay_int * self.delays[1].buffer[0, idx]
        )
        path_2 = rvb.scale_sat_int64_to_int32_floor(path_2)

        path_1 = self.mod_allpasses[0].process_xcore(path_1)
        path_1 = self.delays[0].process_channels_xcore([path_1])[0]
        path_1 = self.lowpasses[1].process_xcore(path_1)
        path_1 = rvb.apply_gain_xcore(path_1, self.decay_int)
        path_1 = self.allpasses[4].process_xcore(path_1)
        path_1 = self.delays[1].process_channels_xcore([path_1])[0]

        path_2 = self.mod_allpasses[1].process_xcore(path_2)
        path_2 = self.delays[2].process_channels_xcore([path_2])[0]
        path_2 = self.lowpasses[2].process_xcore(path_2)
        path_2 = rvb.apply_gain_xcore(path_2, self.decay_int)
        path_2 = self.allpasses[5].process_xcore(path_2)
        path_2 = self.delays[3].process_channels_xcore([path_2])[0]

        # 0.6 * 2 ** 29 = 322122547.2
        # chosen as one of the closest ot the whole number
        # and to give extra headroom for maccs
        scale = 322122547
        scale_q = 29
        output_l = 1 << (scale_q - 1)
        output_l += self.delays[0].buffer[0, self.taps_l[0]] * scale
        output_l += self.delays[0].buffer[0, self.taps_l[1]] * scale
        output_l -= self.allpasses[4]._buffer_int[self.taps_l[2]] * scale
        output_l += self.delays[1].buffer[0, self.taps_l[3]] * scale
        output_l -= self.delays[2].buffer[0, self.taps_l[4]] * scale
        output_l -= self.allpasses[5]._buffer_int[self.taps_l[5]] * scale
        output_l -= self.delays[3].buffer[0, self.taps_l[6]] * scale
        utils.int64(output_l)
        output_l = utils.int32_mult_sat_extract(output_l, 1, scale_q)

        output_r = 1 << (scale_q - 1)
        output_r += self.delays[2].buffer[0, self.taps_r[0]] * scale
        output_r += self.delays[2].buffer[0, self.taps_r[1]] * scale
        output_r -= self.allpasses[5]._buffer_int[self.taps_r[2]] * scale
        output_r += self.delays[3].buffer[0, self.taps_r[3]] * scale
        output_r -= self.delays[0].buffer[0, self.taps_r[4]] * scale
        output_r -= self.allpasses[4]._buffer_int[self.taps_r[5]] * scale
        output_r -= self.delays[1].buffer[0, self.taps_r[6]] * scale
        utils.int64(output_r)
        output_r = utils.int32_mult_sat_extract(output_r, 1, scale_q)

        # move output taps
        for n in range(7):
            self.taps_l[n] += 1
            self.taps_r[n] += 1
            if self.taps_l[n] >= self.tap_lens_l[n]:
                self.taps_l[n] = 0
            if self.taps_r[n] >= self.tap_lens_r[n]:
                self.taps_r[n] = 0

        output_l_final = rvb._2maccs_sat_xcore(output_l, output_r, self.wet_1_int, self.wet_2_int)
        output_l_final = self._effect_gain.process_xcore(output_l_final)
        output_l_final += rvb.apply_gain_xcore(sample_list_int[0], self.dry_int)
        utils.int64(output_l_final)
        output_l_final = utils.saturate_int64_to_int32(output_l_final)

        output_r = rvb._2maccs_sat_xcore(output_r, output_l, self.wet_1_int, self.wet_2_int)
        output_r = self._effect_gain.process_xcore(output_r)
        output_r += rvb.apply_gain_xcore(sample_list_int[1], self.dry_int)
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
