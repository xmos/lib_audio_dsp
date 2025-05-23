# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Cascaded biquads Stages consist of several biquad filters connected
together in series.
"""

from ..design.stage import Stage, find_config
from ..dsp import cascaded_biquads as casc_bq
import numpy as np
from typing import Any


def _parametric_eq_doc(wrapped):
    """Generate docs for parametric eq."""
    import inspect
    from ..dsp import biquad

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
        fc = []
        for bq in self.dsp_block.biquads:
            fc.extend(bq.int_coeffs)
        a = np.array(fc, dtype=np.int32)
        return a

    @_parametric_eq_doc
    def make_parametric_eq(self, filter_spec: list[list[Any]]) -> "CascadedBiquads":
        """Configure this instance as a Parametric Equaliser.

        This allows each of the 8 biquads to be individually designed using the designer
        methods for the biquad. This expects to receive a list of up to 8 biquad design descriptions
        where a biquad design description is of the form::

            ["type", args...]

        where "type" is a string defining how the biquad should be designed e.g. "lowpass", and args...
        is all the parameters to design that type of filter. All options and arguments are listed below::{generated_doc}
        """
        self.details = dict(type="parametric")
        self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, filter_spec)
        return self

    def make_butterworth_highpass(self, N: int, fc: float) -> "CascadedBiquads":
        """Configure this instance as an Nth order Butterworth highpass
        filter using N/2 cascaded biquads.

        For details on the implementation, see
        :class:`audio_dsp.dsp.cascaded_biquads.make_butterworth_highpass`

        Parameters
        ----------
        N : int
            Filter order, must be even
        fc : float
            -3 dB frequency in Hz.
        """
        self.details = dict(type="butterworth highpass", N=N, fc=fc)
        self.dsp_block = casc_bq.butterworth_highpass(self.fs, self.n_in, N, fc)
        return self

    def make_butterworth_lowpass(self, N: int, fc: float) -> "CascadedBiquads":
        """Configure this instance as an Nth order Butterworth lowpass
        filter using N/2 cascaded biquads.

        For details on the implementation, see
        :class:`audio_dsp.dsp.cascaded_biquads.make_butterworth_lowpass`

        Parameters
        ----------
        N : int
            Filter order, must be even
        fc : float
            -3 dB frequency in Hz.
        """
        self.details = dict(type="butterworth lowpass", N=N, fc=fc)
        self.dsp_block = casc_bq.butterworth_lowpass(self.fs, self.n_in, N, fc)
        return self


class CascadedBiquads16(Stage):
    """16 cascaded biquad filters. This allows up to 16 second order
    biquad filters to be run in series.

    This can be used for either:

    - an Nth order filter built out of cascaded second order sections
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

        filter_spec = [
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
            ["bypass"],
        ]
        self.dsp_block = casc_bq.parametric_eq_16band(self.fs, self.n_in, filter_spec)

        self.filter_coeffs = []
        self.left_shift = []
        for bq in self.dsp_block.biquads:
            self.filter_coeffs.extend(bq.coeffs)
            self.left_shift.append(bq.b_shift)

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
    def make_parametric_eq(self, filter_spec: list[list[Any]]) -> "CascadedBiquads16":
        """Configure this instance as a Parametric Equaliser.

        This allows each of the 16 biquads to be individually designed using the designer
        methods for the biquad. This expects to receive a list of up to 8 biquad design descriptions
        where a biquad design description is of the form::

            ["type", args...]

        where "type" is a string defining how the biquad should be designed e.g. "lowpass", and args...
        is all the parameters to design that type of filter. All options and arguments are listed below::{generated_doc}
        """
        self.details = dict(type="parametric")
        self.dsp_block = casc_bq.parametric_eq_16band(self.fs, self.n_in, filter_spec)
        return self
