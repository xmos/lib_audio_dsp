import numpy as np

from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils


class mixer(dspg.dsp_block):
    # add 2 signals but attnuate first to maintain headroom
    def __init__(self, num_channels, gain_db=-6, Q_sig=dspg.Q_SIG):
        super().__init__(Q_sig)
        self.num_channels = num_channels
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
        self.gain_int = utils.int32(self.gain * 2**30)

    def process(self, sample_list):
        scaled_samples = np.array(sample_list)*self.gain
        y = np.sum(scaled_samples)

        return y

    def process_xcore(self, sample_list):
        y = 0
        for sample in sample_list:
            sample_int = utils.int32(round(sample * 2**self.Q_sig))
            scaled_sample = utils.vpu_mult(sample_int, self.gain_int)
            y = utils.int32(y + scaled_sample)

        y_flt = (float(y)*2**-self.Q_sig)

        return y_flt

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

    def process_xcore(self, sample_list):
        sample_int_0 = utils.int32(round(sample_list[0] * 2**self.Q_sig))
        sample_int_1 = utils.int32(round(sample_list[1] * 2**self.Q_sig))

        y = utils.int32(sample_int_0 - sample_int_1)

        y_flt = (float(y)*2**-self.Q_sig)

        return y_flt


class fixed_gain(dspg.dsp_block):
    # multiply every sample by a fixed gain value
    def __init__(self, gain_db, Q_sig=dspg.Q_SIG):
        super().__init__(Q_sig)
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
        self.gain_int = utils.int32(self.gain * 2**30)

    def process(self, sample):
        y = sample*self.gain
        return y

    def process_xcore(self, sample):
        sample_int = utils.int32(round(sample * 2**self.Q_sig))
        y = utils.vpu_mult(sample_int, self.gain_int)

        y_flt = (float(y)*2**-self.Q_sig)

        return y_flt

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
        self.gain_int = utils.int32(self.gain * 2**30)


class switch(dspg.dsp_block):
    def __init__(self, Q_sig=dspg.Q_SIG):
        super().__init__(Q_sig)
        self.switch_position = 0
        return

    def process(self, sample_list):
        y = sample_list[self.switch_position]
        return y

    def process_xcore(self, sample_list):
        return self.process(sample_list)

    def move_switch(self, position):
        self.switch_position = position
        return
