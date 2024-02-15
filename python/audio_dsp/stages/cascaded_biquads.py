# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
from ..dsp import cascaded_biquads as casc_bq
import numpy as np
from typing import Any


class CascadedBiquads(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("cascaded_biquads"), **kwargs)
        self.create_outputs(self.n_in)

        filter_spec = [['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass']]
        self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, filter_spec)

        self.filter_coeffs = []
        self.left_shift = []
        for bq in self.dsp_block.biquads:
            self.filter_coeffs.extend(bq.coeffs)
            self.left_shift.append(bq.b_shift)

        self.set_control_field_cb("filter_coeffs",
                                  lambda: [i for i in self.get_fixed_point_coeffs()])
        self.set_control_field_cb("left_shift",
                                  lambda: [i.b_shift for i in self.dsp_block.biquads])

    def get_fixed_point_coeffs(self):
        fc = []
        for bq in self.dsp_block.biquads:
            fc.extend(bq.coeffs)
        a = np.array(fc)
        return np.array(a*(2**30), dtype=np.int32)

    def make_parametric_eq(self, filter_spec: list[list[Any]]) -> "CascadedBiquads":
        """
        Configure this instance as a Parametric Equaliser.
        """
        self.details = dict(type="parametric")
        self.dsp_block = casc_bq.parametric_eq_8band(self.fs, self.n_in, filter_spec)
        return self
    
    def make_butterworth_highpass(self, N: int, fc: float) -> "CascadedBiquads":
        """
        Configure this instance as a Butterworth highpass filter.
        """
        self.details = dict(type="butterworth lowpass", N=N, fc=fc)
        self.details = dict(type="butterworth highpass", N=N, fc=fc)
        self.dsp_block = casc_bq.butterworth_highpass(self.fs, self.n_in, N, fc)
        return self
    
    def make_butterworth_lowpass(self, N: int, fc: float) -> "CascadedBiquads":
        """
        Configure this instance as a Butterworth lowpass filter.
        """
        self.details = dict(type="butterworth lowpass", N=N, fc=fc)
        self.dsp_block = casc_bq.butterworth_lowpass(self.fs, self.n_in, N, fc)
        return self
