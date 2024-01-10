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

    def make_lowpass(self, f, q):
        self.filt =  bq.biquad_lowpass(self.fs, f, q)

    def make_highpass(self, f, q):
        self.filt =  bq.biquad_highpass(self.fs, f, q)

    def make_bandpass(self, f, bw):
        self.filt =  bq.biquad_bandpass(self.fs, f, bw)

    def make_bandstop(self, f, bw):
        self.filt =  bq.biquad_bandstop(self.fs, f, bw)

    def make_notch(self, f, q):
        self.filt =  bq.biquad_notch(self.fs, f, q)

    def make_allpass(self, f, q):
        self.filt =  bq.biquad_allpass(self.fs, f, q)

    def make_peaking(self, f, q, boost_db):
        self.filt =  bq.biquad_peaking(self.fs, f, q, boost_db)

    def make_constant_q(self, f, q, boost_db):
        self.filt =  bq.biquad_constant_q(self.fs, f, q, boost_db)

    def make_lowshelf(self, f, q, boost_db):
        self.filt =  bq.biquad_lowshelf(self.fs, f, q, boost_db)

    def make_highshelf(self, f, q, boost_db):
        self.filt =  bq.biquad_highshelf(self.fs, f, q, boost_db)

    def make_linkwitz(self, f0, q0, fp, qp):
        self.filt =  bq.biquad_linkwitz(self.fs, f0, q0, fp, qp)
