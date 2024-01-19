from ..design.stage import Stage
import audio_dsp.dsp.cascaded_biquads as casc_bq
import numpy as np

CASCADED_BIQUADS_CONFIG = """
---
module:
  cascaded_biquads:
    left_shift:
      type: int
      size: 8
    filter_coeffs:
      type: int32_t
      size: 40
      attribute: DWORD_ALIGNED
includes:
  - "stdint.h"
  - "stages/adsp_module.h"
"""

class CascadedBiquads(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=CASCADED_BIQUADS_CONFIG, **kwargs)
        self.create_outputs(self.n_in)

        filter_spec = [['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass'],
                  ['bypass']]
        self.filt = casc_bq.parametric_eq(self.fs, filter_spec)

        self.filter_coeffs = []
        self.left_shift = []
        for bq in self.filt.biquads:
            self.filter_coeffs.extend(bq.coeffs)
            self.left_shift.append(bq.b_shift)

        self.set_control_field_cb("filter_coeffs",
                                  lambda: " ".join([str(i) for i in self.get_fixed_point_coeffs()]))
        self.set_control_field_cb("left_shift",
                                  lambda: " ".join([str(i) for i in self.left_shift]))

    def process(self, in_channels):
        """
        Run Biquad on the input channels and return the output

        Args:
            in_channels: list of numpy arrays

        Returns:
            list of numpy arrays.
        """

    def get_fixed_point_coeffs(self):
        a = np.array(self.filter_coeffs.coeffs)
        return np.array(a*(2**30), dtype=np.int32)
