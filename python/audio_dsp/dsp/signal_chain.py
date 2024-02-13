import numpy as np

from audio_dsp.dsp import generic as dspg
from audio_dsp.dsp import utils


class mixer(dspg.dsp_block):
    # add 2 signals but attnuate first to maintain headroom
    def __init__(self, fs, num_channels, gain_db=-6, Q_sig=dspg.Q_SIG):
        super().__init__(fs, num_channels, Q_sig)
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

    def process_frame(self, frame):
        # convert to np array to make taking all channels for the nth sample easy
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process(frame_np[:, sample])

        return [output]

    def process_frame_xcore(self, frame):
        # convert to np array to make taking all channels for the nth sample easy
        frame_np = np.array(frame)
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process_xcore(frame_np[:, sample])

        return [output]

    def freq_response(self, nfft=512):
        # flat response scaled by gain
        w = np.fft.rfftfreq(nfft)
        h = np.ones_like(w)*self.gain
        return w, h


class adder(mixer):
    # just a mixer with no attenuation
    def __init__(self, fs, num_channels):
        super().__init__(fs, num_channels, gain_db=0)


class subtractor(dspg.dsp_block):

    def __init__(self, fs, Q_sig=dspg.Q_SIG):
        # always has 2 channels
        super().__init__(fs, 2, Q_sig)

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

    def process_frame(self, frame):
        # convert to np array to make taking all channels for the nth sample easy
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process([frame[0][sample], frame[1][sample]])

        return [output]

    def process_frame_xcore(self, frame):
        # convert to np array to make taking all channels for the nth sample easy
        frame_size = frame[0].shape[0]
        output = np.zeros(frame_size)
        for sample in range(frame_size):
            output[sample] = self.process_xcore([frame[0][sample], frame[1][sample]])

        return [output]


class fixed_gain(dspg.dsp_block):
    """
    Multiply every sample by a fixed gain value

    In the current implementation, the maximum boost is 6dB.

    """
    def __init__(self, fs, n_chans, gain_db, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)
        assert gain_db < 6, "Maximum fixed gain is +6dB"
        self.gain_db = gain_db
        self.gain = utils.db2gain(gain_db)
        self.gain_int = utils.int32(self.gain * 2**30)

    def process(self, sample, channel=0):
        y = sample*self.gain
        return y

    def process_xcore(self, sample, channel=0):
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
        assert gain_db < 6, "Maximum volume control gain is +6dB"
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
