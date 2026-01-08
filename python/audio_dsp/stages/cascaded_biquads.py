# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Cascaded biquads Stages consist of several biquad filters connected
together in series.
"""

from audio_dsp.design.stage import Stage, find_config
from audio_dsp.dsp import cascaded_biquads as casc_bq
from audio_dsp.models.cascaded_biquads import (
    CascadedBiquadsParameters,
    CascadedBiquads16Parameters,
    NthOrderFilterParameters,
)
import audio_dsp.models.fields as bqm

import numpy as np
from typing import Any


def _parametric_eq_doc(wrapped):
    """Generate docs for parametric eq."""
    import inspect
    from audio_dsp.dsp import biquad

    # find all the biquad design methods
    design_funcs = [
        f[1]
        for f in inspect.getmembers(biquad, inspect.isfunction)
        if f[0].startswith("make_biquad_")
    ]
    assert design_funcs, "Design functions not found!"

    # get arg names excluding fs which is the first arg.
    func_args = [inspect.getfullargspec(f)[0][1:] for f in design_funcs]

    # construct a string describing all the design options.
    doc_gen = "\n\n"
    for f, args in zip(design_funcs, func_args):
        arg_str = ", ".join(['"' + f.__name__.removeprefix("make_biquad_") + '"', *args])
        doc_gen += f"            | [{arg_str}]\n"
    wrapped.__doc__ = wrapped.__doc__.format(generated_doc=doc_gen)

    return wrapped


def _bq_spec_to_parameters(filter_spec: list[list[Any]], out_len=8):
    """Convert a biquad specification to a list of Biquad parameters."""
    filters = []
    for spec in filter_spec:
        if spec[0] == "allpass":
            filters.append(bqm.biquad_allpass(filter_freq=spec[1], q_factor=spec[2]))
        if spec[0] == "bandpass":
            filters.append(bqm.biquad_bandpass(filter_freq=spec[1], bw=spec[2]))
        if spec[0] == "bandstop":
            filters.append(bqm.biquad_bandstop(filter_freq=spec[1], bw=spec[2]))
        if spec[0] == "bypass":
            filters.append(bqm.biquad_bypass())
        if spec[0] == "constant_q":
            filters.append(
                bqm.biquad_constant_q(filter_freq=spec[1], q_factor=spec[2], boost_db=spec[3])
            )
        if spec[0] == "gain":
            filters.append(bqm.biquad_gain(gain_db=spec[1]))
        if spec[0] == "highpass":
            filters.append(bqm.biquad_highpass(filter_freq=spec[1], q_factor=spec[2]))
        if spec[0] == "highshelf":
            filters.append(
                bqm.biquad_highshelf(filter_freq=spec[1], q_factor=spec[2], boost_db=spec[3])
            )
        if spec[0] == "linkwitz":
            filters.append(bqm.biquad_linkwitz(f0=spec[1], q0=spec[2], fp=spec[3], qp=spec[4]))
        if spec[0] == "lowpass":
            filters.append(bqm.biquad_lowpass(filter_freq=spec[1], q_factor=spec[2]))
        if spec[0] == "lowshelf":
            filters.append(
                bqm.biquad_lowshelf(filter_freq=spec[1], q_factor=spec[2], boost_db=spec[3])
            )
        if spec[0] == "mute":
            filters.append(bqm.biquad_mute())
        if spec[0] == "notch":
            filters.append(bqm.biquad_notch(filter_freq=spec[1], q_factor=spec[2]))
        if spec[0] == "peaking":
            filters.append(
                bqm.biquad_peaking(filter_freq=spec[1], q_factor=spec[2], boost_db=spec[3])
            )

    if len(filters) < out_len:
        # pad with bypass filters
        filters.extend([bqm.biquad_bypass() for _ in range(out_len - len(filters))])

    return filters


class CascadedBiquads(Stage):
    """8 cascaded biquad filters. This allows up to 8 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
    - a parametric EQ, where several biquad filters are used at once.

    For documentation on the individual biquad filters, see
    :class:`audio_dsp.stages.biquad.Biquad` and
    :class:`audio_dsp.dsp.biquad.biquad`

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.cascaded_biquad.cascaded_biquad`
        The DSP block class; see :ref:`CascadedBiquads` for
        implementation details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("cascaded_biquads"), **kwargs)
        self.create_outputs(self.n_in)

        filter_spec = [["bypass"] for _ in range(8)]
        self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, filter_spec)

        self.set_control_field_cb(
            "filter_coeffs", lambda: [i for i in self._get_fixed_point_coeffs()]
        )
        self.set_control_field_cb(
            "left_shift", lambda: [i.b_shift for i in self.dsp_block.biquads]
        )

        self.stage_memory_parameters = (self.n_in,)
        self.parameters = CascadedBiquadsParameters()

    def _get_fixed_point_coeffs(self):
        """Get the fixed point coefficients for all biquads."""
        fc = []
        for bq in self.dsp_block.biquads:
            fc.extend(bq.int_coeffs)
        a = np.array(fc, dtype=np.int32)
        return a

    @_parametric_eq_doc
    def make_parametric_eq(self, filter_spec: list[list[Any]]):
        """Configure this CascadedBiquads instance as a Parametric Equaliser based on new
        parameters.

        This allows each of the 8 biquads to be individually designed using the designer
        methods for the biquad. This expects to receive a list of up to 8 biquad design descriptions
        where a biquad design description is of the form::

            ["type", args...]

        where "type" is a string defining how the biquad should be designed e.g. "lowpass", and args...
        is all the parameters to design that type of filter.

        Parameters
        ----------
        filter_spec : list[list[Any]]
            A list of lists, each inner list contains the parameters for
            a single biquad filter. The first element of each inner list
            is the filter type, the remaining elements are the
            parameters for that filter type. The available filter types
            and their parameters are:{generated_doc}
        """
        parameters = CascadedBiquadsParameters(filters=_bq_spec_to_parameters(filter_spec))
        self.set_parameters(parameters)

    def make_butterworth_highpass(self, N: int, fc: float):
        """Configure this CascadedBiquads instance as an Nth order Butterworth highpass
        filter using N/2 cascaded biquads.

        For details on the implementation, see
        :class:`audio_dsp.dsp.cascaded_biquads.make_butterworth_highpass`

        Parameters
        ----------
        N : int
            The order of the filter. Must be even and less than 16.
        fc : float
            The -3dB cutoff frequency in Hz.
        """
        if N not in (2, 4, 6, 8, 10, 12, 14, 16):
            raise ValueError("Order must be one of 2, 4, 6, 8, 10, 12, 14, 16")
        parameters = NthOrderFilterParameters(
            type="highpass", filter="butterworth", order=N, filter_freq=fc
        )
        self.set_parameters(parameters)

    def make_butterworth_lowpass(self, N: int, fc: float):
        """Configure this CascadedBiquads instance as an Nth order Butterworth lowpass
        filter using N/2 cascaded biquads.

        Parameters
        ----------
        N : int
            The order of the filter. Must be even and less than 16.
        fc : float
            The -3dB cutoff frequency in Hz.
        """
        if N not in (2, 4, 6, 8, 10, 12, 14, 16):
            raise ValueError("Order must be one of 2, 4, 6, 8, 10, 12, 14, 16")
        parameters = NthOrderFilterParameters(
            type="lowpass", filter="butterworth", order=N, filter_freq=fc
        )
        self.set_parameters(parameters)

    def set_parameters(self, parameters: CascadedBiquadsParameters | NthOrderFilterParameters):
        """Update the parameters of the CascadedBiquads stage.

        Parameters
        ----------
        parameters : CascadedBiquadsParameters | NthOrderFilterParameters
            The parameters to update the cascaded biquads with.
        """
        self.parameters = parameters
        if isinstance(parameters, CascadedBiquadsParameters):
            model = parameters.model_dump()
            biquads = [list(spec.values()) for spec in model["filters"]]
            self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, biquads)
        else:
            if parameters.type == "bypass":
                self.dsp_block = casc_bq.parametric_eq_8band(
                    self.fs, self.n_in, [["bypass"] for _ in range(8)]
                )
            elif parameters.type == "lowpass" and parameters.filter == "butterworth":
                self.dsp_block = casc_bq.butterworth_lowpass(
                    self.fs, self.n_in, parameters.order, parameters.filter_freq
                )
            elif parameters.type == "highpass" and parameters.filter == "butterworth":
                self.dsp_block = casc_bq.butterworth_highpass(
                    self.fs, self.n_in, parameters.order, parameters.filter_freq
                )
            else:
                raise ValueError(
                    f"Unsupported filter type {parameters.type} or filter {parameters.filter}"
                )


class CascadedBiquads16(Stage):
    """16 cascaded biquad filters. This allows up to 16 second order
    biquad filters to be run in series.

    This can be used for:

    - a parametric EQ, where several biquad filters are used at once.

    For documentation on the individual biquad filters, see
    :class:`audio_dsp.stages.biquad.Biquad` and
    :class:`audio_dsp.dsp.biquad.biquad`

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.cascaded_biquad.cascaded_biquad_16`
        The DSP block class; see :ref:`CascadedBiquads16` for
        implementation details.

    """

    def __init__(self, **kwargs):
        super().__init__(config=find_config("cascaded_biquads_16"), **kwargs)
        self.create_outputs(self.n_in)

        filter_spec = [["bypass"] for _ in range(16)]
        self.dsp_block = casc_bq.parametric_eq_16band(self.fs, self.n_in, filter_spec)

        self.set_control_field_cb(
            "filter_coeffs_lower", lambda: [i for i in self._get_fixed_point_coeffs_lower()]
        )
        self.set_control_field_cb(
            "filter_coeffs_upper", lambda: [i for i in self._get_fixed_point_coeffs_upper()]
        )
        self.set_control_field_cb(
            "left_shift", lambda: [i.b_shift for i in self.dsp_block.biquads]
        )

        self.stage_memory_parameters = (self.n_in,)
        self.parameters = CascadedBiquads16Parameters()

    def _get_fixed_point_coeffs_lower(self):
        fc = []
        for bq in self.dsp_block.biquads[:8]:
            fc.extend(bq.int_coeffs)
        a = np.array(fc, dtype=np.int32)
        return a

    def _get_fixed_point_coeffs_upper(self):
        fc = []
        for bq in self.dsp_block.biquads[8:]:
            fc.extend(bq.int_coeffs)
        a = np.array(fc, dtype=np.int32)
        return a

    @_parametric_eq_doc
    def make_parametric_eq(self, filter_spec: list[list[Any]]):
        """Configure this CascadedBiquads16 instance as a Parametric Equaliser based on new
        parameters.

        This allows each of the 16 biquads to be individually designed using the designer
        methods for the biquad. This expects to receive a list of up to 8 biquad design descriptions
        where a biquad design description is of the form::

            ["type", args...]

        where "type" is a string defining how the biquad should be designed e.g. "lowpass", and args...
        is all the parameters to design that type of filter.

        Parameters
        ----------
        filter_spec : list[list[Any]]
            A list of lists, each inner list contains the parameters for
            a single biquad filter. The first element of each inner list
            is the filter type, the remaining elements are the
            parameters for that filter type. The available filter types
            and their parameters are:{generated_doc}
        """
        parameters = CascadedBiquads16Parameters(
            filters=_bq_spec_to_parameters(filter_spec, out_len=16)
        )
        self.set_parameters(parameters)

    def set_parameters(self, parameters: CascadedBiquads16Parameters):
        """Update the parameters of the CascadedBiquads16 stage.

        Parameters
        ----------
        parameters : CascadedBiquads16Parameters
            The parameters to update the cascaded biquads with.
        """
        self.parameters = parameters
        model = parameters.model_dump()
        biquads = [[*spec.values()] for spec in model["filters"]]
        self.dsp_block = casc_bq.parametric_eq_16band(self.fs, self.n_in, biquads)


class ParametricEq8b(CascadedBiquads):
    """An 8 band parametric equalizer stage. This stage allows up to 8 biquad
    filters to be run in series. Each filter can be configured
    independently.

    For documentation on the individual biquad filters, see
    :class:`audio_dsp.stages.biquad.Biquad` and
    :class:`audio_dsp.dsp.biquad.biquad`

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.cascaded_biquad.cascaded_biquad`
        The DSP block class; see :ref:`CascadedBiquads` for
        implementation details.
    """

    def set_parameters(self, parameters: CascadedBiquadsParameters):  # pyright: ignore
        """Update the parameters of the ParametricEq8b stage.

        Parameters
        ----------
        parameters : CascadedBiquadsParameter
            A list of BiquadParameters to update the cascaded biquads with.
        """
        super().set_parameters(parameters)


class ParametricEq16b(CascadedBiquads16):
    """A 16 band parametric equalizer stage. This stage allows up to 16 biquad
    filters to be run in series. Each filter can be configured
    independently.

    For documentation on the individual biquad filters, see
    :class:`audio_dsp.stages.biquad.Biquad` and
    :class:`audio_dsp.dsp.biquad.biquad`

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.cascaded_biquad.cascaded_biquad_16`
        The DSP block class; see :ref:`CascadedBiquads16` for
        implementation details.
    """

    def set_parameters(self, parameters: CascadedBiquads16Parameters):
        """Update the parameters of the ParametricEq16b stage."""
        self.parameters = parameters
        model = parameters.model_dump()
        biquads = [[*spec.values()] for spec in model["filters"]]
        self.dsp_block = casc_bq.parametric_eq_16band(self.fs, self.n_in, biquads)


class NthOrderFilter(CascadedBiquads):
    """An Nth order filter stage. This stage allows up a 16th order filter
    to be created by cascading 8 second order biquad filters.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.cascaded_biquad.cascaded_biquad`
        The DSP block class; see :ref:`CascadedBiquads` for
        implementation details.
    """

    def set_parameters(self, parameters: NthOrderFilterParameters):  # pyright: ignore
        """Update the parameters of the NthOrderFilter stage.

        Parameters
        ----------
        parameters : NthOrderFilter
            The parameters to update the Nth order filter with.
        """
        super().set_parameters(parameters)
