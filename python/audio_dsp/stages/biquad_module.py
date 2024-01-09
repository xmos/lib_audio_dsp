from ..design.stage import Stage
import audio_dsp.dsp.biquad as bq
import numpy as np

# Copied from Shuchita's branch of usb audio
BIQUAD_CONFIG = """
---
module:
  biquad:
    left_shift:
      type: int32_t
    filter_coeffs:
      type: int32_t
      size: "5"
      attribute: DWORD_ALIGNED
    reserved:
      type: int32_t
      size: "3"
includes:
  - "stdint.h"
  - "adsp_module.h"
"""

class BiquadModule(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=BIQUAD_CONFIG, **kwargs)
        self.create_outputs(self.n_in)
        self.fs = 48000
        self.filt = bq.biquad_lowpass(self.fs, 1000, 0.7)
        self.set_control_field_cb("filter_coeffs",
                                  lambda: " ".join([str(i) for i in self.get_fixed_point_coeffs()]))
        self.set_control_field_cb("left_shift",
                                  lambda: str(self.filt.b_shift))

    def process(self, in_channels):
        """
        Run Biquad on the input channels and return the output

        Args:
            in_channels: list of numpy arrays

        Returns:
            list of numpy arrays.
        """

    def get_fixed_point_coeffs(self):
        a = np.array(self.filt.coeffs)
        return np.array(a*(2**30), dtype=np.int32)

