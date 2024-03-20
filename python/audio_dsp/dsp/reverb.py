import audio_dsp.dsp.generic as dspg
import soundfile as sf
from pathlib import Path
import os
import numpy as np

import audio_dsp.dsp.utils as utils

class allpass_fv(dspg.dsp_block):
    def __init__(self, delay, feedback_gain):
        self.delay = delay
        self._feedback = feedback_gain
        self.buffer = np.zeros(self.delay)
        self.buffer_idx = 0
        
    def process(self, sample):

        buff_out = self.buffer[self.buffer_idx]

        output = -sample + buff_out
        self.buffer[self.buffer_idx] = sample + (buff_out*self._feedback)
        
        self.buffer_idx += 1
        if self.buffer_idx >= self.delay:
            self.buffer_idx = 0

        return output


class comb_fv(dspg.dsp_block):
    def __init__(self, delay, feedback_gain, damping):
        self._delay = delay
        self._feedback = feedback_gain
        self._buffer = np.zeros(self._delay)
        self.buffer_idx = 0
        self.filterstore = 0
        self.damp1 = damping
        self.damp2 = 1 - self.damp1
        
    def process(self, sample):

        output = self._buffer[self.buffer_idx]

        self.filterstore = (output*self.damp2) + (self.filterstore*self.damp1)

        self._buffer[self.buffer_idx] = sample + (self.filterstore*self._feedback)
        
        self.buffer_idx += 1
        if self.buffer_idx >= self._delay:
            self.buffer_idx = 0

        return output


class freeverb(dspg.dsp_block):
    def __init__(self, room_size=2, decay=0.5, damping=0.4, wet_gain_db=-1, dry_gain_db=-1):
        """_summary_

        Parameters
        ----------
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
        self.damping = damping
        self.feedback = decay*0.28 + 0.7
        self.wet = utils.db2gain(wet_gain_db)
        self.dry = utils.db2gain(dry_gain_db)
        self.gain = 0.015
        self.room_size = room_size
        self.comb_lengths = (np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])*self.room_size).astype(int)
        self.ap_lengths = (np.array([556, 441, 341, 225])*self.room_size).astype(int)

        self.combs = [comb_fv(self.comb_lengths[0], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[1], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[2], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[3], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[4], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[5], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[6], self.feedback, self.damping),
                      comb_fv(self.comb_lengths[7], self.feedback, self.damping)]

        self.allpasses = [allpass_fv(self.ap_lengths[0], 0.5),
                          allpass_fv(self.ap_lengths[1], 0.5),
                          allpass_fv(self.ap_lengths[2], 0.5),
                          allpass_fv(self.ap_lengths[3], 0.5)]

    def get_buffer_lens(self):
        total_buffers = 0
        for cb in self.combs:
            total_buffers += cb._delay
        for ap in self.allpasses:
            total_buffers += ap.delay
        return total_buffers

    def process(self, sample):

        output = 0
        input = sample*self.gain
        for cb in self.combs:
            output += cb.process(input)

        for ap in self.allpasses:
            output = ap.process(output)

        output = output*self.wet + sample*self.dry
        return output


if __name__ == "__main__":
    hydra_audio_path = os.environ['hydra_audio_PATH']
    filepath = Path(hydra_audio_path, 'acoustic_team_test_audio',
                    'speech', "010_male_female_single-talk_seq.wav")
    sig, fs = sf.read(filepath)

    reverb = freeverb()
    print(reverb.get_buffer_lens())
    
    output = np.zeros_like(sig)
    for n in range(len(sig)):
        output[n] = reverb.process(sig[n])
    
    sf.write('reverb_out.wav', output, fs)