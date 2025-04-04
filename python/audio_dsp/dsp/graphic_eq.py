# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import audio_dsp.dsp.generic as dspg
import audio_dsp.dsp.biquad as bq
import numpy as np
import audio_dsp.dsp.utils as utils
from copy import deepcopy


Q_GEQ = 29 # allow +12 dB


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
            raise ValueError("Sample rate too low for 10 band graphic EQ")
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

        # compensation for sum of all bands
        shift = -0.37

        self.gain_offset = gain_offset
        self.gains_db = gains_db
        self.biquads = []

        for f in range(10):
            coeffs = bq.make_biquad_bandpass(fs, cfs[f], bw[f])
            coeffs = bq._apply_biquad_gain(coeffs, gains[f] + shift)
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
        self.gains = [utils.db2gain(x + self.gain_offset) for x in self.gains_db]
        self.gains_int = [utils.float_to_int32(x, Q_GEQ) for x in self.gains]

    def _get_coeffs(self):
        all_coeffs = []
        for biquad in self.biquads:
            all_coeffs.extend(biquad.int_coeffs)
        return all_coeffs

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
        h_all = h_all**2 * self.gains[0]
        for n in range(1, 10):
            _, h = self.biquads[n].freq_response(nfft)
            h_all += ((-1) ** n) * h**2 * self.gains[n]

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
        y = utils.int64(1 << (Q_GEQ - 1))
        for n in range(10):
            this_band = utils.float_to_fixed(sample, self.Q_sig)
            this_band = self.biquads[n].process_xcore(this_band, 2 * channel)
            this_band = self.biquads[n].process_xcore(this_band, 2 * channel + 1)
            if n % 2 != 0:
                this_band = -this_band
            y += utils.int64(this_band * self.gains_int[n])

        y = utils.int32_mult_sat_extract(y, 1, Q_GEQ)
        y_flt = utils.fixed_to_float(y, self.Q_sig)

        return y_flt

def _print_coeffs():
    for fs in [(48000 + 44100) / 2, (96000 + 88200) / 2, 192000]:
        this_geq = graphic_eq_10_band(fs, 1, np.zeros(10))
        print(f"// coeffs for {fs} Hz")
        coeffs_arr = []
        lsh_arr = []
        str_arr = ""
        for n in range(10):
            these_coeffs = this_geq.biquads[n].int_coeffs
            coeffs_arr.append(these_coeffs)
            str_arr += ', '.join(str(c) for c in these_coeffs)
            str_arr += ",\n "
            lsh_arr.append(this_geq.biquads[n].b_shift)
        print("int32_t coeffs[50] = {" + str_arr + "}\n")

    assert np.all(lsh_arr==0), "Error, not all lsh equal to zero"

if __name__ == "__main__":
    _print_coeffs()