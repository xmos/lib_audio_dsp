# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Biquad Stages can be used for basic audio filters."""

from audio_dsp.design.stage import Stage, find_config
import audio_dsp.dsp.biquad as bq
import numpy as np

from audio_dsp.models.biquad import BiquadParameters, BiquadSlewParameters
import audio_dsp.models.fields as fields


def _ws(locals):
    """
    Without self.

    Parameters
    ----------
    locals : dict
        a dictionary

    Returns
    -------
    dict
        l with the entry "self" removed
    """
    return {k: v for k, v in locals.items() if k != "self"}


class Biquad(Stage):
    """
    A second order biquadratic filter, which can be used to make many
    common second order filters. The filter is initialised in a
    bypass state, and the ``make_*`` methods can be used to calculate the
    coefficients.

    This Stage implements a direct form 1 biquad filter:
    ``a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]``

    For efficiency the biquad coefficients are normalised by ``a0`` and the
    output ``a`` coefficients multiplied by -1.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.biquad.biquad`
        The DSP block class; see :ref:`Biquad`
        for implementation details.
    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("biquad"), **kwargs)
        if self.fs is None:
            raise ValueError("Biquad requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.parameters = BiquadParameters()
        self.dsp_block: bq = bq.biquad(bq.make_biquad_bypass(self.fs), self.fs, self.n_in)
        self.set_control_field_cb("filter_coeffs", self._get_fixed_point_coeffs)
        self.set_control_field_cb("left_shift", lambda: self.dsp_block.b_shift)
        self.stage_memory_parameters = (self.n_in,)

    def _get_fixed_point_coeffs(self) -> list[int]:
        return self.dsp_block.int_coeffs

    def make_bypass(self) -> "Biquad":
        """Make this biquad a bypass by setting the b0 coefficient to
        1.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_bypass())
        self.set_parameters(parameters)
        return self

    def make_gain(self, gain_db: float) -> "Biquad":
        """Make this biquad a gain stage.

        Parameters
        ----------
        gain_db : float
            Gain of the filter in decibels.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_gain(gain_db=gain_db))
        self.set_parameters(parameters)
        return self

    def make_lowpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad a second order low pass filter.

        Parameters
        ----------
        f : float
            Cutoff frequency of the filter in Hz.
        q : float
            Q factor of the filter roll-off. 0.707 is equivalent to a
            Butterworth response.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_lowpass(filter_freq=f, q_factor=q))
        self.set_parameters(parameters)
        return self

    def make_highpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad a second order high pass filter.

        Parameters
        ----------
        f : float
            Cutoff frequency of the filter in Hz.
        q : float
            Q factor of the filter roll-off. 0.707 is equivalent to a
            Butterworth response.
        """
        parameters = BiquadParameters(
            filter_type=fields.biquad_highpass(filter_freq=f, q_factor=q)
        )
        self.set_parameters(parameters)
        return self

    def make_bandpass(self, f: float, bw: float) -> "Biquad":
        """Make this biquad a second order bandpass filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        bw : float
            Bandwidth of the filter in octaves.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_bandpass(filter_freq=f, bw=bw))
        self.set_parameters(parameters)
        return self

    def make_bandstop(self, f: float, bw: float) -> "Biquad":
        """Make this biquad a second order bandstop filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        bw : float
            Bandwidth of the filter in octaves.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_bandstop(filter_freq=f, bw=bw))
        self.set_parameters(parameters)
        return self

    def make_notch(self, f: float, q: float) -> "Biquad":
        """Make this biquad a notch filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_notch(filter_freq=f, q_factor=q))
        self.set_parameters(parameters)
        return self

    def make_allpass(self, f: float, q: float) -> "Biquad":
        """Make this biquad an all pass filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        """
        parameters = BiquadParameters(filter_type=fields.biquad_allpass(filter_freq=f, q_factor=q))
        self.set_parameters(parameters)
        return self

    def make_peaking(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a peaking filter.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        parameters = BiquadParameters(
            filter_type=fields.biquad_peaking(filter_freq=f, q_factor=q, boost_db=boost_db)
        )
        self.set_parameters(parameters)
        return self

    def make_constant_q(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a peaking filter with constant Q.

        Constant Q means that the bandwidth of the filter remains
        constant as the gain varies. It is commonly used for graphic
        equalisers.

        Parameters
        ----------
        f : float
            Center frequency of the filter in Hz.
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        parameters = BiquadParameters(
            filter_type=fields.biquad_constant_q(filter_freq=f, q_factor=q, boost_db=boost_db)
        )
        self.set_parameters(parameters)
        return self

    def make_lowshelf(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a second order low shelf filter.

        The Q factor is defined in a similar way to standard low pass,
        i.e. > 0.707 will yield peakiness (where the shelf response does
        not monotonically change). The level change at f will be
        boost_db/2.

        Parameters
        ----------
        f : float
            Cutoff frequency of the shelf in Hz, where the gain is
            boost_db/2
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        parameters = BiquadParameters(
            filter_type=fields.biquad_lowshelf(filter_freq=f, q_factor=q, boost_db=boost_db)
        )
        self.set_parameters(parameters)
        return self

    def make_highshelf(self, f: float, q: float, boost_db: float) -> "Biquad":
        """Make this biquad a second order high shelf filter.

        The Q factor is defined in a similar way to standard high pass,
        i.e. > 0.707 will yield peakiness (where the shelf response does
        not monotonically change). The level change at f will be
        boost_db/2.

        Parameters
        ----------
        f : float
            Cutoff frequency of the shelf in Hz, where the gain is
            boost_db/2
        q : float
            Q factor of the filter.
        boost_db : float
            Gain of the filter in decibels.
        """
        parameters = BiquadParameters(
            filter_type=fields.biquad_highshelf(filter_freq=f, q_factor=q, boost_db=boost_db)
        )
        self.set_parameters(parameters)
        return self

    def make_linkwitz(self, f0: float, q0: float, fp: float, qp: float) -> "Biquad":
        """Make this biquad a Linkwitz Transform biquad filter.

        The Linkwitz Transform changes the low frequency cutoff of a filter, and is commonly used to change the low
        frequency roll off slope of a loudspeaker. When applied to a
        loudspeaker, it will change the cutoff frequency from f0 to fp,
        and the Q factor from q0 to qp.

        Parameters
        ----------
        f0 : float
            The original cutoff frequency of the filter in Hz.
        q0 : float
            The original quality factor of the filter at f0.
        fp : float
            The target cutoff frequency for the filter in Hz.
        qp : float
            The target quality factor for the filter.
        """
        parameters = BiquadParameters(
            filter_type=fields.biquad_linkwitz(f0=f0, q0=q0, fp=fp, qp=qp)
        )
        self.set_parameters(parameters)
        return self

    def set_parameters(self, parameters: BiquadParameters):
        """Set biquad filter parameters.

        Args:
            parameters: New biquad parameters to apply
        """
        # If parameters is a dict, convert to BiquadParameters
        if isinstance(parameters, dict):
            parameters = BiquadParameters(**parameters)

        filter_type = parameters.filter_type

        # Call the appropriate make_* method based on the filter type
        if filter_type.type == "lowpass":
            new_coeffs = bq.make_biquad_lowpass(
                self.fs, filter_type.filter_freq, filter_type.q_factor
            )
        elif filter_type.type == "highpass":
            new_coeffs = bq.make_biquad_highpass(
                self.fs, filter_type.filter_freq, filter_type.q_factor
            )
        elif filter_type.type == "bandpass":
            new_coeffs = bq.make_biquad_bandpass(self.fs, filter_type.filter_freq, filter_type.bw)
        elif filter_type.type == "bandstop":
            new_coeffs = bq.make_biquad_bandstop(self.fs, filter_type.filter_freq, filter_type.bw)
        elif filter_type.type == "notch":
            new_coeffs = bq.make_biquad_notch(
                self.fs, filter_type.filter_freq, filter_type.q_factor
            )
        elif filter_type.type == "allpass":
            new_coeffs = bq.make_biquad_allpass(
                self.fs, filter_type.filter_freq, filter_type.q_factor
            )
        elif filter_type.type == "peaking":
            new_coeffs = bq.make_biquad_peaking(
                self.fs, filter_type.filter_freq, filter_type.q_factor, filter_type.boost_db
            )
        elif filter_type.type == "constant_q":
            new_coeffs = bq.make_biquad_constant_q(
                self.fs, filter_type.filter_freq, filter_type.q_factor, filter_type.boost_db
            )
        elif filter_type.type == "lowshelf":
            new_coeffs = bq.make_biquad_lowshelf(
                self.fs, filter_type.filter_freq, filter_type.q_factor, filter_type.boost_db
            )
        elif filter_type.type == "highshelf":
            new_coeffs = bq.make_biquad_highshelf(
                self.fs, filter_type.filter_freq, filter_type.q_factor, filter_type.boost_db
            )
        elif filter_type.type == "linkwitz":
            new_coeffs = bq.make_biquad_linkwitz(
                self.fs, filter_type.f0, filter_type.q0, filter_type.fp, filter_type.qp
            )
        elif filter_type.type == "gain":
            new_coeffs = bq.make_biquad_gain(self.fs, filter_type.gain_db)
        elif filter_type.type == "bypass":
            new_coeffs = bq.make_biquad_bypass(self.fs)
        else:
            raise ValueError(f"Unknown filter type: {filter_type.type}")

        self.dsp_block.update_coeffs(new_coeffs)

        # Store the parameters
        self.parameters = parameters


class BiquadSlew(Biquad):
    """
    A second order biquadratic filter with slew, which can be used to
    make many common second order filters. The filter is initialised in a
    bypass state, and the ``make_*`` methods can be used to calculate the
    coefficients. This variant will slew between filter coefficients when
    they are changed.

    This Stage implements a direct form 1 biquad filter:
    ``a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]``

    For efficiency the biquad coefficients are normalised by ``a0`` and the
    output ``a`` coefficients multiplied by -1.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.biquad.biquad_slew`
        The DSP block class; see :ref:`BiquadSlew`
        for implementation details.
    """

    def __init__(self, **kwargs):
        Stage.__init__(self, config=find_config("biquad_slew"), **kwargs)
        if self.fs is None:
            raise ValueError("Biquad slew requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.parameters = BiquadParameters()
        init_coeffs = bq.make_biquad_bypass(self.fs)
        self.dsp_block: bq = bq.biquad_slew(init_coeffs, self.fs, self.n_in)
        self.set_control_field_cb("filter_coeffs", self._get_fixed_point_coeffs)
        self.set_control_field_cb("left_shift", lambda: self.dsp_block.b_shift)
        self.set_control_field_cb("slew_shift", lambda: self.dsp_block.slew_shift)
        self.stage_memory_parameters = (self.n_in,)

    def _get_fixed_point_coeffs(self) -> list[int]:
        return self.dsp_block.target_coeffs_int

    def set_slew_shift(self, slew_shift):
        """Set the slew shift for a biquad object. This sets how fast the
        filter will slew between filter coefficients.
        """
        self.dsp_block.slew_shift = slew_shift

    def set_parameters(self, parameters: BiquadSlewParameters):  #pyright: ignore
        """Set the slewing biquad parameters."""
        self.dsp_block.slew_shift = parameters.slew_shift
        super().set_parameters(parameters)
