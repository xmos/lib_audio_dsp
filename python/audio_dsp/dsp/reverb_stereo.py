import audio_dsp.dsp.reverb as rv
import audio_dsp.dsp.generic as dspg
import numpy as np
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.utils as utils
from copy import deepcopy
import warnings


class reverb_room_stereo(rv.reverb_room):
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
    def wet_db(self):
        """The gain applied to the wet signal in dB."""
        return self._wet_db

    @wet_db.setter
    def wet_db(self, x):
        if x > 0:
            warnings.warn(f"Wet gain {x} saturates to 0 dB", UserWarning)
            x = 0

        self._wet_db = x
        self._wet = utils.db2gain(x)
        self.wet_1 = self.wet * (self.width / 2 + 0.5)
        self.wet_2 = self.wet * ((1 - self.width) / 2)

        self.wet_1_int = rv.float_to_q_verb(self.wet_1)
        self.wet_2_int = rv.float_to_q_verb(self.wet_2)

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

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        # recalculate wet gains
        self.wet_db = self.wet_db

    def process(self, sample):
        raise NotImplementedError

    def process_xcore(self, sample):
        raise NotImplementedError

    def process_channels(self, sample_list: list[float]):
        """
        Add reverberation to a signal, using floating point maths.

        Take one new sample and return the sample with reverb.
        Input should be scaled with 0 dB = 1.0.

        """
        reverb_input = (sample_list[0] + sample_list[1]) * self.pregain
        reverb_input = self._predelay.process_channels([reverb_input])

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
        sample_list_int = utils.float_list_to_int32(sample_list, self.Q_sig)
        # TODO avoid overflow
        reverb_input = sample_list_int[0] + sample_list_int[1]
        reverb_input = rv.apply_gain_xcore(reverb_input, self.pregain_int)
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

        output_l_final = rv.apply_gain_xcore(output_l, self.wet_1_int)
        output_l_final += rv.apply_gain_xcore(output_r, self.wet_2_int)
        output_l_final = self._effect_gain.process_xcore(output_l_final)
        output_l_final += rv.apply_gain_xcore(sample_list_int[0], self.dry_int)
        utils.int64(output_l_final)
        output_l_final = utils.saturate_int64_to_int32(output_l_final)

        output_r = rv.apply_gain_xcore(output_r, self.wet_1_int)
        output_r += rv.apply_gain_xcore(output_l, self.wet_2_int)
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
        self.dry_db = utils.db(np.sqrt((1 - mix) * np.cos(omega)))
        self.wet_db = utils.db(np.sqrt(mix * np.sin(omega)))
        # there's an extra gain of 10 dB added to the wet channel to
        # make it similar level to the dry, so that the mixing is smooth.
        # Couldn't add it to the wet gain itself as it's in q31
