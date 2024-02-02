import numpy as np
from audio_dsp.dsp import utils as utils

Q_SIG = 27
HEADROOM_BITS = 31-Q_SIG
HEADROOM_DB = utils.db(2**HEADROOM_BITS)


class dsp_block():
    """
    Generic DSP block, all blocks should inherit from this class and implement
    it's methods.

    Parameters
    ----------
    fs : int
        sampling frequency in Hz.
    Q_sig: int, optional
        Q format of the signal, number of bits after the decimal point.
        Defaults to Q27.

    Attributes
    ----------
    fs : int
        sampling frequency in Hz.
    Q_sig: int
        Q format of the signal, number of bits after the decimal point.
    """

    def __init__(self, fs, n_chans, Q_sig=Q_SIG):
        self.fs = fs
        self.n_chans = n_chans
        self.Q_sig = Q_sig
        return

    def process(self, sample: float):
        """
        Take 1 new sample and give it back. Do no processing for the generic
        block.
        """
        return sample

    def process_int(self, sample: float):
        """
        Take 1 new sample and return 1 processed sample.

        For the generic implementation, scale and quantize the input, call the
        float implementation, then scale back to 1.0 = 0dB.
        """
        sample_int = utils.int32(sample * 2**self.Q_sig)
        y = self.process(float(sample_int))
        y_flt = float(y)*2**-self.Q_sig

        return y_flt

    def process_frame(self, frame):
        # simple multichannel, assumes no channel unique states!
        n_outputs = frame.shape[0]
        frame_size = frame.shape[1]
        output = np.zeros_like(frame)
        for chan in range(n_outputs):
            for sample in range(frame_size):
                output[chan, sample] = self.process(frame[chan, sample],
                                                    channel=chan)

        return output

    def process_frame_int(self, frame):
        # simple multichannel, but integer. Assumes no channel unique states!
        n_outputs = frame.shape[0]
        frame_size = frame.shape[1]
        output = np.zeros_like(frame)
        for chan in range(n_outputs):
            for sample in range(frame_size):
                output[chan, sample] = self.process_int(frame[chan, sample],
                                                        channel=chan)

        return output

    def freq_response(self, nfft=512):
        """
        The frequency response of the module for a nominal input.

        The generic module has a flat frequency response.
        """
        f = np.fft.rfftfreq(nfft)*self.fs
        h = np.ones_like(f)
        return f, h
