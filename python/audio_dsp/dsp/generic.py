import numpy as np
from audio_dsp.dsp import utils as utils

Q_SIG = 27
HEADROOM_BITS = 31-Q_SIG
HEADROOM_DB = utils.db(2**HEADROOM_BITS)


class dsp_block():
    def __init__(self, Q_sig=Q_SIG):
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

    def freq_response(self, nfft=512):
        # generic module has a flat response
        w = np.fft.rfftfreq(nfft)
        h = np.ones_like(w)
        return w, h
