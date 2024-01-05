from ..design.stage import Stage

# Copied from Shuchita's branch of usb audio
PEQ_CONFIG = """
---
module:
  parametric_eq:
    filter_coeffs:
      type: int32_t
      size: FILTERS * DSP_NUM_COEFFS_PER_BIQUAD
      attribute: DWORD_ALIGNED
includes:
  - stdint.h
  - "dsp.h"
  - "adsp_module.h"
defines:
  FILTERS: 4
"""

class ParametricEq(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=PEQ_CONFIG, **kwargs)
        self.create_outputs(self.n_in)

        self.xhistory = [[0, 0] for _ in range(self.n_in)]
        self.yhistory = [[0, 0] for _ in range(self.n_in)]

    def process(self, in_channels):
        """
        Run peq on the input channels and return the output

        Args:
            in_channels: list of numpy arrays

        Returns:
            list of numpy arrays.
        """


