import numpy as np

from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils


class mixer(dspg.dsp_block):
    # add 2 signals but attnuate first to maintain headroom
    def __init__(self, num_channels, gain_db=-6):
        self.num_channels = num_channels
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)

    def process(self, sample_list):
        scaled_samples = np.array(sample_list)*self.gain
        y = np.sum(scaled_samples)

        return y

    def freq_response(self, nfft=512):
        # flat response scaled by gain
        w = np.fft.rfftfreq(nfft)
        h = np.ones_like(w)*self.gain
        return w, h


class adder(mixer):
    # just a mixer with no attenuation
    def __init__(self, num_channels):
        super.__init__(self, num_channels, db_gain=0)


class subtractor(dspg.dsp_block):
    # subtract 1st input from the second
    def process(self, sample_list):
        y = sample_list[0] - sample_list[1]

        return y


class fixed_gain(dspg.dsp_block):
    # multiply every sample by a fixed gain value
    def __init__(self, gain_db):
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)

    def process(self, sample):
        y = sample*self.gain
        return y

    def freq_response(self, nfft=512):
        # flat response scaled by gain
        w = np.fft.rfftfreq(nfft)
        h = np.ones_like(w)*self.gain
        return w, h


class volume_control(fixed_gain):
    # just a fixed gain with an exposed set gain
    def set_gain(self, gain_db):
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
