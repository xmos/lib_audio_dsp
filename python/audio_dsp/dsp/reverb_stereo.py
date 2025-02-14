# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""DSP blocks for stereo reverb effects."""

import audio_dsp.dsp.reverb as rv
import audio_dsp.dsp.generic as dspg
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.utils as utils
import warnings
import audio_dsp.dsp.reverb_base as rvb


class reverb_room_stereo(rvb.reverb_stereo_base):
    """Generate a stereo room reverb effect. This is based on Freeverb by
    Jezar at Dreampoint. Each channel consists of 8 parallel comb filters fed
    into 4 series all-pass filters, and the reverberator outputs are mixed
    according to the ``width`` parameter.

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
        width=1.0,
        wet_gain_db=-1,
        dry_gain_db=-1,
        pregain=0.0075,
        predelay=10,
        max_predelay=None,
        Q_sig=dspg.Q_SIG,
    ):
        assert n_chans == 2, f"Stereo reverb only supports 2 channel. {n_chans} specified"

        # initalise wet/dry gains, width, and predelay
        super().__init__(
            fs, n_chans, width, wet_gain_db, dry_gain_db, pregain, predelay, max_predelay, Q_sig
        )

        self._effect_gain = sc.fixed_gain(fs, n_chans, 10)

        # the magic freeverb delay line lengths are for 44.1kHz, so
        # scale them with sample rate and room size
        default_comb_lengths = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])
        default_ap_lengths = np.array([556, 441, 341, 225])
        default_spread = 23

        # buffer lengths
        length_scaling = self.fs / 44100 * max_room_size
        self.comb_lengths = (default_comb_lengths * length_scaling).astype(int)
        self.ap_lengths = (default_ap_lengths * length_scaling).astype(int)
        self.spread_length = int(default_spread * length_scaling)

        # feedbacks
        init_fb = 0.5
        init_damping = 0.4
        self.combs_l = [
            rv.comb_fv(self.comb_lengths[0], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[1], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[2], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[3], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[4], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[5], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[6], init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[7], init_fb, init_damping),
        ]

        self.combs_r = [
            rv.comb_fv(self.comb_lengths[0] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[1] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[2] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[3] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[4] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[5] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[6] + self.spread_length, init_fb, init_damping),
            rv.comb_fv(self.comb_lengths[7] + self.spread_length, init_fb, init_damping),
        ]

        feedback_ap = 0.5
        self.allpasses_l = [
            rv.allpass_fv(self.ap_lengths[0], feedback_ap),
            rv.allpass_fv(self.ap_lengths[1], feedback_ap),
            rv.allpass_fv(self.ap_lengths[2], feedback_ap),
            rv.allpass_fv(self.ap_lengths[3], feedback_ap),
        ]

        self.allpasses_r = [
            rv.allpass_fv(self.ap_lengths[0] + self.spread_length, feedback_ap),
            rv.allpass_fv(self.ap_lengths[1] + self.spread_length, feedback_ap),
            rv.allpass_fv(self.ap_lengths[2] + self.spread_length, feedback_ap),
            rv.allpass_fv(self.ap_lengths[3] + self.spread_length, feedback_ap),
        ]

        # set filter delays
        self.decay = decay
        self.damping = damping
        self.room_size = room_size

    def reset_state(self):
        """Reset all the delay line values to zero."""
        for cb in self.combs_l:
            cb.reset_state()
        for cb in self.combs_r:
            cb.reset_state()
        for ap in self.allpasses_l:
            ap.reset_state()
        for ap in self.allpasses_r:
            ap.reset_state()
        self._predelay.reset_state()

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
        ret = float(self.combs_l[0].feedback)
        return ret

    @feedback.setter
    def feedback(self, x):
        for n in range(len(self.combs_l)):
            self.combs_l[n].set_feedback(x)
            self.combs_r[n].set_feedback(x)

        self.feedback_int = self.combs_r[0].feedback_int

    @property
    def damping(self):
        """How much high frequency attenuation in the room, between 0 and 1."""
        return self.combs_l[0].damp1

    @damping.setter
    def damping(self, x):
        if not (0 <= x <= 1):
            bad_x = x
            x = np.clip(x, 0, 1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)
        for n in range(len(self.combs_l)):
            self.combs_l[n].set_damping(x)
            self.combs_r[n].set_damping(x)

        self.damping_int = self.combs_l[0].damp1_int

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
        spread_delay = int(self.spread_length * self._room_size)

        for n in range(len(self.combs_l)):
            self.combs_l[n].set_delay(comb_delays[n])
            self.combs_r[n].set_delay(comb_delays[n] + spread_delay)

        for n in range(len(self.allpasses_l)):
            self.allpasses_l[n].set_delay(ap_delays[n])
            self.allpasses_r[n].set_delay(ap_delays[n] + spread_delay)

    def process_channels(self, sample_list: list[float]):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        reverb_input = (sample_list[0] + sample_list[1]) * self.pregain
        reverb_input = self._predelay.process_channels([reverb_input])[0]

        output_l = 0
        output_r = 0
        for n in range(len(self.combs_l)):
            output_l += self.combs_l[n].process(reverb_input)
            output_r += self.combs_r[n].process(reverb_input)

        for n in range(len(self.allpasses_l)):
            output_l = self.allpasses_l[n].process(output_l)
            output_r = self.allpasses_r[n].process(output_r)

        output_l_final = output_l * self.wet_1 + output_r * self.wet_2
        output_l_final = self._effect_gain.process(output_l_final) + sample_list[0] * self.dry

        output_r = output_r * self.wet_1 + output_l * self.wet_2
        output_r = self._effect_gain.process(output_r) + sample_list[1] * self.dry

        return [output_l_final, output_r]

    def process_channels_xcore(self, sample_list: list[float]):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        sample_list_int = [utils.float_to_fixed_signal(x, self.Q_sig) for x in sample_list]

        acc = 1 << (rvb.Q_VERB - 1)
        acc += sample_list_int[0] * self.pregain_int
        acc += sample_list_int[1] * self.pregain_int
        utils.int64(acc)
        reverb_input = utils.int32_mult_sat_extract(acc, 1, rvb.Q_VERB)
        reverb_input = self._predelay.process_channels_xcore([reverb_input])[0]

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

        output_l_flt = utils.fixed_to_float_signal(output_l_final, self.Q_sig)
        output_r_flt = utils.fixed_to_float_signal(output_r, self.Q_sig)

        return [output_l_flt, output_r_flt]
