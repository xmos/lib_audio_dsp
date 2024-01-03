from ..design.stage import Stage

# Copied from Shuchita's branch of usb audio
AGC_CONFIG = """
---
module:
  agc:
    gain:
      type: float
includes:
  - stdint.h
  - "dspt_module.h"
"""

class AGC(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=AGC_CONFIG, **kwargs)
        self.create_outputs(self.n_in)


    def process(self, in_channels):
        """
        Run AGC on the input channels and return the output

        Args:
            in_channels: list of numpy arrays

        Returns:
            list of numpy arrays.
        """

