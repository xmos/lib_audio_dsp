from ..design.stage import Stage, find_config
from ..dsp import cascaded_biquads as casc_bq
import numpy as np


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
        self.filt = casc_bq.parametric_eq_8band(self.fs, filter_spec)

        self.filter_coeffs = []
        self.left_shift = []
        for bq in self.filt.biquads:
            self.filter_coeffs.extend(bq.coeffs)
            self.left_shift.append(bq.b_shift)

        self.set_control_field_cb("filter_coeffs",
                                  lambda: [i for i in self.get_fixed_point_coeffs()])
        self.set_control_field_cb("left_shift",
                                  lambda: [i.b_shift for i in self.filt.biquads])

    def process(self, in_channels):
        """
        Run Biquad on the input channels and return the output

        Args:
            in_channels: list of numpy arrays

        Returns:
            list of numpy arrays.
        """

    def get_fixed_point_coeffs(self):
        fc = []
        for bq in self.filt.biquads:
            fc.extend(bq.coeffs)
        a = np.array(fc)
        return np.array(a*(2**30), dtype=np.int32)

    def make_parametric_eq(self, filter_spec):
        self.details = dict(type="parametric")
        self.filt = casc_bq.parametric_eq_8band(self.fs, filter_spec)
        return self
    
    def make_butterworth_highpass(self, N, fc):
        self.details = dict(type="butterworth highpass", N=N, fc=fc)
        self.filt = casc_bq.butterworth_highpass(self.fs, N, fc)
        return self
    
    def make_butterworth_lowpass(self, N, fc):
        self.details = dict(type="butterworth lowpass", N=N, fc=fc)
        self.filt = casc_bq.butterworth_lowpass(self.fs, N, fc)
        return self
