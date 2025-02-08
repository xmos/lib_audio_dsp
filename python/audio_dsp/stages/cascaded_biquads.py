# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Cascaded biquad Stages consist of several biquad filters connected
together in series.
"""

from typing import Annotated, Any, Literal

import numpy as np
from annotated_types import Len
from pydantic import BaseModel, Field, RootModel, create_model
from pydantic.json_schema import SkipJsonSchema

import audio_dsp.stages.biquad as bq
from audio_dsp.design.stage import Stage, find_config
from audio_dsp.dsp import cascaded_biquads as casc_bq
from audio_dsp.models.cascaded_biquads import ParametricEqParameters


def _parametric_eq_doc(wrapped):
    """Generate docs for parametric eq."""
    import inspect

    from ..dsp import biquad

    # find all the biquad design methods
    design_funcs = [
        f[1] for f in inspect.getmembers(biquad, inspect.isfunction) if f[0].startswith("biquad_")
    ]

    # get arg names excluding fs and num channels which are the first 2 args.
    func_args = [inspect.getfullargspec(f)[0][2:] for f in design_funcs]

    # construct a string describing all the design options.
    doc_gen = "\n\n"
    for f, args in zip(design_funcs, func_args):
        arg_str = ", ".join(['"' + f.__name__ + '"', *args])
        doc_gen += f"            [{arg_str}]\n"
    wrapped.__doc__ = wrapped.__doc__.format(generated_doc=doc_gen)
    return wrapped


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

        filter_spec = [
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
        ]
        self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, filter_spec)

        self.filter_coeffs = []
        self.left_shift = []
        for bq in self.dsp_block.biquads:
            self.filter_coeffs.extend(bq.coeffs)
            self.left_shift.append(bq.b_shift)

        self.set_control_field_cb(
            "filter_coeffs", lambda: [i for i in self._get_fixed_point_coeffs()]
        )
        self.set_control_field_cb(
            "left_shift", lambda: [i.b_shift for i in self.dsp_block.biquads]
        )

        self.stage_memory_parameters = (self.n_in,)

    def _get_fixed_point_coeffs(self):
        """Get the fixed point coefficients for all biquads."""
        coeffs = []
        for bq in self.dsp_block.biquads:
            coeffs.extend(bq.coeffs)
        return coeffs

    @_parametric_eq_doc
    def make_parametric_eq(self, filter_spec: list[list[Any]]) -> "CascadedBiquads":
        """Update parametric eq configuration based on new parameters.

        Parameters
        ----------
        filter_spec : list[list[Any]]
            A list of lists, each inner list contains the parameters for
            a single biquad filter. The first element of each inner list
            is the filter type, the remaining elements are the
            parameters for that filter type. The available filter types
            and their parameters are:{generated_doc}

        Returns
        -------
        CascadedBiquads
            self
        """
        self.details = dict(filter_spec=filter_spec)
        self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, filter_spec)
        return self

    def make_butterworth_highpass(self, N: int, fc: float) -> "CascadedBiquads":
        """Update parametric eq configuration to be a butterworth highpass filter.

        Parameters
        ----------
        N : int
            The order of the filter. Must be even and less than 16.
        fc : float
            The cutoff frequency in Hz.

        Returns
        -------
        CascadedBiquads
            self
        """
        self.details = dict(N=N, fc=fc)
        self.dsp_block = casc_bq.butterworth_highpass(self.fs, self.n_in, N, fc)
        return self

    def make_butterworth_lowpass(self, N: int, fc: float) -> "CascadedBiquads":
        """Update parametric eq configuration to be a butterworth lowpass filter.

        Parameters
        ----------
        N : int
            The order of the filter. Must be even and less than 16.
        fc : float
            The cutoff frequency in Hz.

        Returns
        -------
        CascadedBiquads
            self
        """
        self.details = dict(N=N, fc=fc)
        self.dsp_block = casc_bq.butterworth_lowpass(self.fs, self.n_in, N, fc)
        return self


class ParametricEq(CascadedBiquads):
    """A parametric equalizer stage. This stage allows up to 8 biquad
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

    def set_parameters(self, parameters: ParametricEqParameters):
        model = parameters.model_dump()
        biquads = [[*spec.values()] for spec in model["filters"]]
        self.make_parametric_eq(biquads)
