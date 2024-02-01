import numpy as np
from audio_dsp.dsp import utils as utils

Q_SIG = 27
HEADROOM_BITS = 31-Q_SIG
HEADROOM_DB = utils.db(2**HEADROOM_BITS)


class dsp_block():
    def __init__(self, fs, n_chans, Q_sig=Q_SIG):
        self.fs = fs
        self.n_chans = n_chans
        self.Q_sig = Q_sig
        return

    def process(self, sample):

        return sample

    def process_int(self, sample):
        # lazy int implementation by scaling to int, hen calling double precision
        # implementation
        sample_int = np.round(sample * 2**self.Q_sig).astype(np.int32)
        y = self.process(sample_int.astype(np.double))
        y_flt = (y.astype(np.double)*2**-self.Q_sig).astype(np.double)

        return y_flt

    def process_frame(self, frame):
        # simple multichannel, assumes no channel unique states!
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = frame.copy()
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process(this_chan[sample],
                                                 channel=chan)

        return output

    def process_frame_int(self, frame):
        # simple multichannel, but integer. Assumes no channel unique states!
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = frame.copy()
        for chan in range(n_outputs):
            this_chan = output[chan]
            for sample in range(frame_size):
                this_chan[sample] = self.process_int(this_chan[sample],
                                                     channel=chan)

        return output

    def freq_response(self, nfft=512):
        # generic module has a flat response
        f = np.fft.rfftfreq(nfft)*self.fs
        h = np.ones_like(f)
        return f, h
