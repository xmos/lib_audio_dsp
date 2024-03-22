import audio_dsp.dsp.generic as dspg
import soundfile as sf
from pathlib import Path
import os
import numpy as np

import audio_dsp.dsp.utils as utils

Q_VERB = 31


def apply_gain_xcore(sample, gain):
    """Apply the gain to a sample usign fixed-point math, assumes that gain is in Q_alpha format"""
    acc = 1 << (Q_VERB - 1)
    acc += sample * gain
    y = utils.int32_mult_sat_extract(acc, 1, Q_VERB)
    return y


class allpass_fv(dspg.dsp_block):
    def __init__(self, max_delay, starting_delay, feedback_gain):
        # max delay cannot be changed, or you'll overflow the buffer
        self._max_delay = max_delay
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay

        self.delay = starting_delay
        self.feedback = feedback_gain
        self.feedback_int = utils.int32(self.feedback * 2**Q_VERB)
        self._buffer_idx = 0

    def set_delay(self, delay):
        if delay < self._max_delay:
            self.delay = delay
        else:
            self.delay = self._max_delay
            Warning("Delay cannot be greater than max delay, setting to max delay")
        return

    def process(self, sample):

        buff_out = self._buffer[self._buffer_idx]

        output = -sample + buff_out
        self._buffer[self.buffer_idx] = sample + (buff_out*self.feedback)
        
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output

    def process_xcore(self, sample_int):
        assert isinstance(sample_int, int), "Input sample must be an integer"

        buff_out = self._buffer_int[self._buffer_idx]

        output = utils.int32(-sample_int + buff_out)
        self._buffer_int[self._buffer_idx] = sample_int 
        self._buffer_int[self._buffer_idx] += apply_gain_xcore(buff_out, self.feedback_int)
        # check for overflow
        utils.int32(self._buffer_int[self._buffer_idx])

        # move buffer head
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output


class comb_fv(dspg.dsp_block):
    def __init__(self, max_delay, starting_delay, feedback_gain, damping):
        # max delay cannot be changed, or you'll overflow the buffer
        self._max_delay = max_delay
        self._buffer = np.zeros(self._max_delay)
        self._buffer_int = [0] * self._max_delay

        self.delay = starting_delay
        self.feedback = feedback_gain
        self.feedback_int = utils.int32(self.feedback * 2**Q_VERB)

        self._buffer_idx = 0
        self._filterstore = 0
        self._filterstore_int = 0
        self.damp1 = damping
        self.damp2 = 1 - self.damp1
        self.damp1_int = utils.int32(self.damp1 * 2**Q_VERB)
        self.damp2_int = utils.int32(2**31 - 1 - self.damp1_int)

    def set_delay(self, delay):
        if delay < self._max_delay:
            self.delay = delay
        else:
            self.delay = self._max_delay
            Warning("Delay cannot be greater than max delay, setting to max delay")
        return

    def process(self, sample):

        output = self._buffer[self._buffer_idx]

        self._filterstore = (output*self.damp2) + (self._filterstore*self.damp1)

        self._buffer[self.buffer_idx] = sample + (self._filterstore*self._feedback)
        
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output

    def process_xcore(self, sample_int):
        assert isinstance(sample_int, int), "Input sample must be an integer"

        output = self._buffer_int[self._buffer_idx]

        self._filterstore_int = apply_gain_xcore(self._filterstore_int, self.damp1_int)
        self._filterstore_int += apply_gain_xcore(output, self.damp2_int)
        utils.int32(self._filterstore_int)

        self._buffer_int[self._buffer_idx] = sample_int 
        self._buffer_int[self._buffer_idx] += apply_gain_xcore(self._filterstore_int, self.feedback_int)
        utils.int32(self._buffer_int[self._buffer_idx])
        
        self._buffer_idx += 1
        if self._buffer_idx >= self.delay:
            self._buffer_idx = 0

        return output

class freeverb(dspg.dsp_block):
    def __init__(self, fs, n_chans, max_room_size=1, room_size=0.5, decay=0.5, damping=0.4, wet_gain_db=-1, dry_gain_db=-1, Q_sig=dspg.Q_SIG,):
        """A room reverb effect based on Freeverb by Jezar at
        Dreampoint.

        Parameters
        ----------
        max_room_size : float, optional
            sets the maximum size of the delay buffers, can only be set
            at initialisation
        room_size : float, optional
            how big the room is, sets delay line lengths. Likely between
            0 and 1, unless you have a huge room
        decay : int, optional
            how long the reverberation of the room is, between 0 and 1
        damping : float, optional
           how much high frequency attenuation in the room
        wet_gain_db : int, optional
            wet signal gain
        dry_gain_db : int, optional
            dry signal gain
        """
        super().__init__(fs, 1, Q_sig)

        self.damping = damping
        self.feedback = decay*0.28 + 0.7  # avoids too much or too little feedback
        self.wet = utils.db2gain(wet_gain_db)
        self.dry = utils.db2gain(dry_gain_db)

        self.wet_int = utils.int32(self.wet * 2**Q_VERB)
        self.dry_int = utils.int32(self.dry * 2**Q_VERB)

        # pregain going into the reverb
        self.gain = 0.03125
        self.out_gain = 0.015/self.gain
        self.wet *= self.out_gain
        self.wet_int = utils.int32(self.wet * 2**Q_VERB)

        self.gain_int = utils.int32(self.gain * 2**Q_VERB)

        if room_size > 1:
            raise ValueError("room_size must be less than 1. For larger rooms, increase max_room size")
        self.room_size = room_size

        # the magic freeverb delay line lengths are for 44.1kHz, so
        # scale them with sample rate and room size
        default_comb_lengths = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])
        default_ap_lengths = np.array([556, 441, 341, 225])

        # buffer lengths
        self.length_scaling = self.fs/44100 * max_room_size
        self.comb_lengths = (default_comb_lengths*self.length_scaling).astype(int)
        self.ap_lengths = (default_ap_lengths*self.length_scaling).astype(int)

        # buffer delays (always < buffer lengths)
        comb_delays = (self.comb_lengths * self.room_size).astype(int)
        ap_delays = (self.ap_lengths * self.room_size).astype(int)

        self.combs = [comb_fv(self.comb_lengths[0], comb_delays[0], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[1], comb_delays[1], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[2], comb_delays[2], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[3], comb_delays[3], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[4], comb_delays[4], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[5], comb_delays[5], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[6], comb_delays[6], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[7], comb_delays[7], self.feedback, self.damping)]

        self.allpasses = [allpass_fv(self.ap_lengths[0], ap_delays[0], 0.5),
                          allpass_fv(self.ap_lengths[1], ap_delays[1], 0.5),
                          allpass_fv(self.ap_lengths[2], ap_delays[2], 0.5),
                          allpass_fv(self.ap_lengths[3], ap_delays[3], 0.5)]

    def get_buffer_lens(self):
        total_buffers = 0
        for cb in self.combs:
            total_buffers += cb._max_delay
        for ap in self.allpasses:
            total_buffers += ap._max_delay
        return total_buffers

    def set_room_size(self, room_size):
        if room_size > 1:
            raise ValueError("room_size must be less than 1. For larger rooms, increase max_room size")
        self.room_size = room_size

        comb_delays = (self.comb_lengths * self.room_size).astype(int)
        ap_delays = (self.ap_lengths * self.room_size).astype(int)

        for n in range(len(self.combs)):
            self.combs[n].set_delay(comb_delays[n])
        
        for n in range(len(self.allpasses)):
            self.allpasses[n].set_delay(ap_delays[n])

        return


    def process(self, sample):

        output = 0
        reverb_input = sample*self.gain
        for cb in self.combs:
            output += cb.process(reverb_input)

        for ap in self.allpasses:
            output = ap.process(output)

        output = output*self.wet + sample*self.dry
        return output

    def process_xcore(self, sample):

        sample_int = utils.int32(round(sample * 2**self.Q_sig))

        output = 0

        reverb_input = apply_gain_xcore(sample_int, self.gain_int)

        for cb in self.combs:
            output += cb.process_xcore(reverb_input)

        for ap in self.allpasses:
            output = ap.process_xcore(output)

        output = apply_gain_xcore(output, self.wet_int)
        output += apply_gain_xcore(sample_int, self.dry_int)
        utils.int32(output)

        return (float(output) * 2**-self.Q_sig)


if __name__ == "__main__":
    # hydra_audio_path = os.environ['hydra_audio_PATH']
    # filepath = Path(hydra_audio_path, 'acoustic_team_test_audio',
    #                 'speech', "010_male_female_single-talk_seq.wav")
    filepath = Path(r"C:\Users\allanskellett\Documents\014_ACM\T-REC-P.501-202005-I!!SOFT-ZST-E\Speech signals\Test Signals Clause 7\Test_Signals_Clause 7\Speech Test Signals Clause 7.3 & 7.4\English_FB_clause_7.3\FB_male_female_single-talk_seq.wav")
    sig, fs = sf.read(filepath)

    sig = sig[:fs*5]
    sig = sig/np.max(np.abs(sig))
    sig = sig* (2**31 - 1)/(2**31)

    reverb = freeverb(fs, 1, room_size=1, decay=1.0, damping=0.5, Q_sig=31)
    print(reverb.get_buffer_lens())
    
    output = np.zeros_like(sig)
    for n in range(len(sig)//2):
        output[n] = reverb.process_xcore(sig[n])

    reverb.set_room_size(0.5)

    for n in range(len(sig)//2):
        output[n + len(sig)//2] = reverb.process_xcore(sig[n + len(sig)//2])

    sf.write('reverb_out.wav', output, fs)