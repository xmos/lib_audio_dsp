import audio_dsp.dsp.biquad as bq
import numpy as np
import audio_dsp.dsp.utils as utils


class graphic_eq_10_band(dspg.dsp_block):
    """
    A 10 band graphic equaliser, with octave spaced center frequencies.

    The equaliser is implemented as a set of parallel 4th order bandpass
    filters. The center frequencies are:
    [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]. Due to the
    nature of the bandpass filters, frequencies below 25 Hz and above
    19 kHz are filtered out. The filter coefficients have been hand
    tuned for common sample rates to minimise ripple in the combined
    output. As with analog graphic equalisers, interactions between
    neighbouring bands means the frequency response is not guaranteed to
    be equal to the slider positions.

    The frequency response ripple with all the gains set to the same
    level is +/- 0.2 dB

    Parameters
    ----------
    gain_offset : float
        Shifts the gains_db values by a number of decibels, by default -12dB
        to allow for an expected gains_db range of -12 to +12 dB.
        This means that setting gains_db to -gains_offset (+12dB) will
        not result in clipping, with the compromise that a gains_db value
        of 0dB will actually reduce the signal level by gain_offset (-12 dB).
    """

    def __init__(self, fs, n_chans, gains_db, gain_offset=-12, Q_sig=dspg.Q_SIG):
        super().__init__(fs, n_chans, Q_sig)

        if fs < 44000:
            raise ValueError("10")
        elif fs < (48000 + 88200) / 2:
            # hand tuned values for 44.1k and 48k
            cfs = [31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8150, 15000]
            bw = [1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 0.75875]
            gains = [-0.3, -0.225, 0.1750, 0, 0.01, 0, 0.05, 0.025, -0.41, -0.25]

        elif fs < (96000 + 176400) / 2:
            # hand tuned values for 88.2k and 96k
            cfs = [31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            bw = [1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.1]
            gains = [-0.3, -0.225, 0.1750, 0, 0.01, 0, 0.0, 0.0, -0.35, -0.2]
        else:
            # hand tuned values for 192k
            cfs = [31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            bw = [1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.1]
            gains = [-0.3, -0.225, 0.1750, 0, 0.01, 0, 0.0, -0.05, -0.3, -0.2]

        self.gains_db = gains_db
        self.biquads = []

        for f in range(10):
            coeffs = bq.make_biquad_bandpass(fs, cfs[f], bw[f])
            coeffs = bq._apply_biquad_gain(coeffs, gains[f])
            # need extra channels as we use each filter twice
            self.biquads.append(bq.biquad(coeffs, fs, n_chans=2 * self.n_chans))

    @property
    def gains_db(self):
        """A list of the gains in decibels for each frequency band. This
        must be a list with 10 values.
        """
        return self._gains_db

    @gains_db.setter
    def gains_db(self, gains_db):
        assert len(gains_db) == 10, "10 gains required for a 10 band EQ"
        self._gains_db = deepcopy(gains_db)
        self.gain_offset = gain_offset
        self.gains = utils.db2gain(self.gains_db - self.gain_offset)

    def freq_response(self, nfft=512):
        """
        Calculate the frequency response of the graphic equaliser.

        The biquad filter coefficients for each biquad are scaled and
        returned to numerator and denominator coefficients, before being
        passed to `scipy.signal.freqz` to calculate the frequency
        response.

        The stages are then combined by summing the complex
        frequency responses with alternating polarities.

        Parameters
        ----------
        nfft : int
            The number of points to compute in the frequency response,
            by default 512.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            A tuple containing the frequency vector and the complex
            frequency response.

        """
        f, h_all = self.biquads[0].freq_response(nfft)
        h_all = h_all**2
        for n in range(1, 10):
            _, h = self.biquads[n].freq_response(nfft)
            h_all += ((-1) ** n) * h**2

        return f, h_all

    def process(self, sample, channel=0):
        """Process the input sample through the 10 band graphic
        equaliser using floating point maths.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on.

        Returns
        -------
        float
            The processed output sample.
        """
        y = 0.0
        for n in range(10):
            this_band = sample
            this_band = self.biquads[n].process(this_band, 2 * channel)
            this_band = self.biquads[n].process(this_band, 2 * channel + 1)
            if n % 2 != 0:
                this_band = -this_band
            y += this_band * self.gains[n]
        return y

    def process_xcore(self, sample, channel=0):
        """Process the input sample through the 10 band graphic
        equaliser using fixed point maths.

        Parameters
        ----------
        sample : float
            The input sample to be processed.
        channel : int
            The channel index to process the sample on.

        Returns
        -------
        float
            The processed output sample.
        """
        y = 0
        for n in range(10):
            this_band = sample
            this_band = self.biquads[n].process_xcore(this_band, 2 * channel)
            this_band = self.biquads[n].process_xcore(this_band, 2 * channel + 1)
            if n % 2 != 0:
                this_band = -this_band
            y += this_band * self.gains[n]
        return y
