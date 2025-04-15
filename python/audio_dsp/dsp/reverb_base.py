# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Base classes and functions for reverb effects."""

import audio_dsp.dsp.generic as dspg
import audio_dsp.dsp.signal_chain as sc
import audio_dsp.dsp.utils as utils
import warnings
import numpy as np
from copy import deepcopy

Q_VERB = 31

# biggest number that is less than 1
_LESS_THAN_1 = ((2**Q_VERB) - 1) / (2**Q_VERB)


def _2maccs_sat_xcore(in1, in2, gain1, gain2):
    acc = 1 << (Q_VERB - 1)
    acc += in1 * gain1
    acc += in2 * gain2
    utils.int64(acc)
    y = utils.int32_mult_sat_extract(acc, 1, Q_VERB)
    return y


def float_to_q_verb(x):
    """Convert a floating point number to Q_VERB format. The input must
    be between 0 and 1. As Q_VERB is typically Q31, care must be taken
    to not overflow by scaling 1.0f*(2**31).
    """
    if x > 1 or x < 0:
        raise ValueError("input must be between 0 and 1")

    if x == 1:
        x_int = utils.int32(2**31 - 1)
    elif x == 0:
        x_int = 0
    else:
        x_int = utils.int32(x * (2**Q_VERB))

    return x_int


def apply_gain_xcore(sample, gain):
    """Apply the gain to a sample using fixed-point math. Assumes that gain is in Q_VERB format."""
    acc = 1 << (Q_VERB - 1)
    acc += sample * gain
    utils.int64(acc)
    y = utils.int32_mult_sat_extract(acc, 1, Q_VERB)
    return y


def scale_sat_int64_to_int32_floor(val):
    """Quanitze an int64 to int32, saturating and quantizing to zero
    in the process. This is useful for feedback paths, where limit
    cycles can occur if you don't round to zero.
    """
    # force the comb filter/all pass feedback to converge to zero and
    # avoid limit noise by rounding to zero. Above 0, truncation does
    # this, but below 0 we truncate to -inf, so add just under 1 to go
    # up instead.
    if val < 0:
        val += (1 << Q_VERB) - 1
        utils.int64(val)

    # saturate
    if val > ((1 << (31 + Q_VERB)) - 1):
        warnings.warn("Saturation occurred", utils.SaturationWarning)
        val = (1 << (31 + Q_VERB)) - 1
    elif val < -(1 << (31 + Q_VERB)):
        warnings.warn("Saturation occurred", utils.SaturationWarning)
        val = -(1 << (31 + Q_VERB))

    # shift to int32
    y = utils.int32(val >> Q_VERB)

    return y


class reverb_base(dspg.dsp_block):
    """
    The base reverb class, containing pre-delay, wet-dry mix and gains.

    Parameters
    ----------
    wet_gain_db : int, optional
        wet signal gain, less than 0 dB.
    dry_gain_db : int, optional
        dry signal gain, less than 0 dB.
    pregain : float, optional
        the amount of gain applied to the signal before being passed
        into the reverb, less than 1. If the reverb raises an
        OverflowWarning, this value should be reduced until it does not.
        The default value of 0.015 should be sufficient for most Q27
        signals.
    predelay : float, optional
        the delay applied to the wet channel in ms.
    max_predelay : float, optional
        the maximum predelay in ms.

    Attributes
    ----------
    pregain : float
    pregain_int : int
        The pregain applied before the reverb as a fixed point number.
    wet_db : float
    wet : float
    wet_int : int
        The linear gain applied to the wet signal as a fixed point
        number.
    dry : float
    dry_db : float
    dry_int : int
        The linear gain applied to the dry signal as a fixed point
        number.
    predelay : float
    """

    def __init__(
        self,
        fs,
        n_chans,
        wet_gain_db=-3,
        dry_gain_db=-3,
        pregain=0.005,
        predelay=10,
        max_predelay=None,
        Q_sig=dspg.Q_SIG,
    ):
        super().__init__(fs, n_chans, Q_sig)

        # predelay
        if max_predelay == None:
            if predelay == 0:
                # single sample delay line
                max_predelay = 1 / fs * 1000
            else:
                max_predelay = predelay

        # single channel delay line, as input is shared
        self._predelay = sc.delay(fs, 1, max_predelay, predelay, "ms")

        # gains
        self.pregain = pregain
        self.wet_db = wet_gain_db
        self.dry_db = dry_gain_db

    @property
    def dry_db(self):
        """The gain applied to the dry signal in dB."""
        return utils.db(self.dry)

    @dry_db.setter
    def dry_db(self, x):
        if x > 0:
            warnings.warn(f"Dry gain {x} saturates to 0 dB", UserWarning)
            x = 0

        self.dry = utils.db2gain(x)

    @property
    def dry(self):
        """The linear gain applied to the dry signal."""
        return self._dry

    @dry.setter
    def dry(self, x):
        self._dry = x
        self.dry_int = float_to_q_verb(self.dry)

    @property
    def wet_db(self):
        """The gain applied to the wet signal in dB."""
        return utils.db(self.wet)

    @wet_db.setter
    def wet_db(self, x):
        if x > 0:
            warnings.warn(f"Wet gain {x} saturates to 0 dB", UserWarning)
            x = 0

        self.wet = utils.db2gain(x)

    @property
    def wet(self):
        """The linear gain applied to the wet signal."""
        return self._wet

    @wet.setter
    def wet(self, x):
        self._wet = x
        self.wet_int = float_to_q_verb(self.wet)

    @property
    def pregain(self):
        """
        The pregain applied before the reverb as a floating point
        number.
        """
        return self._pregain

    @pregain.setter
    def pregain(self, x):
        if not (0 <= x < 1):
            bad_x = x
            x = np.clip(x, 0, _LESS_THAN_1)
            warnings.warn(f"Pregain {bad_x} saturates to {x}", UserWarning)

        self._pregain = x
        self.pregain_int = utils.int32(x * 2**Q_VERB)

    def set_wet_dry_mix(self, mix):
        """
        Will mix wet and dry signal by adjusting wet and dry gains.
        So that when the mix is 0, the output signal is fully dry,
        when 1, the output signal is fully wet. Tries to maintain a
        stable signal level using -4.5 dB Pan Law.

        Parameters
        ----------
        mix : float
            The wet/dry mix, must be [0, 1].
        """
        if not (0 <= mix <= 1):
            bad_mix = mix
            mix = np.clip(mix, 0, 1)
            warnings.warn(f"Wet/dry mix {bad_mix} saturates to {mix}", UserWarning)
        # get an angle [0, pi /2]
        omega = mix * np.pi / 2

        # -4.5 dB
        self.dry = np.sqrt((1 - mix) * np.cos(omega))
        self.wet = np.sqrt(mix * np.sin(omega))
        # there's an extra gain of 10 dB added to the wet channel to
        # make it similar level to the dry, so that the mixing is smooth.
        # Couldn't add it to the wet gain itself as it's in q31


class reverb_stereo_base(reverb_base):
    """
    The base stereo reverb class, containing stereo width. This inherits
    other parameters from reverb_base.

    Parameters
    ----------
    width : float, optional
        how much stereo separation there is between the left and
        right channels. Setting width to 0 will yield a mono signal,
        whilst setting width to 1 will yield the most stereo
        separation.

    Attributes
    ----------
    width : float

    """

    def __init__(
        self,
        fs,
        n_chans,
        width=1.0,
        wet_gain_db=-3,
        dry_gain_db=-3,
        pregain=0.005,
        predelay=10,
        max_predelay=None,
        Q_sig=dspg.Q_SIG,
    ):
        assert n_chans == 2, f"Stereo reverb only supports 2 channel. {n_chans} specified"
        self._width = width
        super().__init__(
            fs, n_chans, wet_gain_db, dry_gain_db, pregain, predelay, max_predelay, Q_sig
        )

    @property
    def wet(self):
        """The linear gain applied to the wet signal."""
        return self._wet

    # override wet setter to also set wet_1 and wet_2
    @wet.setter
    def wet(self, x):
        self._wet = x
        self.wet_1 = self.wet * (self.width / 2 + 0.5)
        self.wet_2 = self.wet * ((1 - self.width) / 2)

        self.wet_1_int = float_to_q_verb(self.wet_1)
        self.wet_2_int = float_to_q_verb(self.wet_2)

    @property
    def width(self):
        """Stereo separation of the reverberated signal."""
        return self._width

    @width.setter
    def width(self, value):
        if not (0 <= value <= 1):
            raise ValueError("width must be between 0 and 1.")
        self._width = value
        # recalculate wet gains
        self.wet = self.wet

    def process(self, sample, channel=0):
        """Process is not implemented for the stereo reverb, as it needs
        2 channels at once.
        """
        raise NotImplementedError

    def process_xcore(self, sample, channel=0):
        """process_xcore is not implemented for the stereo reverb, as it needs
        2 channels at once.
        """
        raise NotImplementedError

    def process_channels(self, sample_list: list[float]):
        """Process_channels is should be implemented for the stereo reverb,
        but depends on the algorithm.

        Parameters
        ----------
        sample_list : list[float]
            A list of the input samples

        Returns
        -------
        list[float]
            The processed samples.
        """
        raise NotImplementedError

    def process_channels_xcore(self, sample_list: list[float]):
        """Process_channels is should be implemented for the stereo reverb,
        but depends on the algorithm.

        Parameters
        ----------
        sample_list : list[float]
            A list of the input samples.

        Returns
        -------
        list[float]
            The processed samples.
        """
        raise NotImplementedError
