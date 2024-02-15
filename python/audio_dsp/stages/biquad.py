# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from ..design.stage import Stage, find_config
import audio_dsp.dsp.biquad as bq
import numpy as np

def _ws(l):
    """
    without self

    Parameters
    ----------
    l : dict
        a dictionary

    Returns
    -------
    dict
        l with the entry "self" removed
    """
    return {k: v for k, v in l.items() if k != "self"}

class Biquad(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("biquad"), **kwargs)
        if self.fs is None:
            raise ValueError("Biquad requires inputs with a valid fs")
        self.fs = int(self.fs)
        self.create_outputs(self.n_in)
        self.set_control_field_cb("filter_coeffs",
                                  lambda: [i for i in self.get_fixed_point_coeffs()])
        self.set_control_field_cb("left_shift",
                                  lambda: self.dsp_block.b_shift)
        self.make_bypass()

    def get_fixed_point_coeffs(self) -> np.ndarray:
        a = np.array(self.dsp_block.coeffs)
        return np.array(a*(2**30), dtype=np.int32)

    def make_bypass(self) -> "Biquad":
        """
        Make this biquad a bypass.
        """
        self.details = {}
        self.dsp_block =  bq.biquad_bypass(self.fs, self.n_in)
        return self

    def make_lowpass(self, f: float, q: float) -> "Biquad":
        """
        Make this biquad a lowpass.
        """
        self.details = dict(type="low pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_lowpass(self.fs, self.n_in, f, q)
        return self

    def make_highpass(self, f: float, q: float) -> "Biquad":
        """
        Make this biquad a highpass.
        """
        self.details = dict(type="high pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_highpass(self.fs, self.n_in, f, q)
        return self

    def make_bandpass(self, f: float, bw: float) -> "Biquad":
        """
        Make this biquad a bandpass.
        """
        self.details = dict(type="band pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_bandpass(self.fs, self.n_in, f, bw)
        return self

    def make_bandstop(self, f: float, bw: float) -> "Biquad":
        """
        Make this biquad a bandstop.
        """
        self.details = dict(type="band stop", **_ws(locals()))
        self.dsp_block =  bq.biquad_bandstop(self.fs, self.n_in, f, bw)
        return self

    def make_notch(self, f: float, q: float) -> "Biquad":
        """
        Make this biquad a notch.
        """
        self.details = dict(type="notch", **_ws(locals()))
        self.dsp_block =  bq.biquad_notch(self.fs, self.n_in, f, q)
        return self

    def make_allpass(self, f: float, q: float) -> "Biquad":
        """
        Make this biquad an allpass.
        """
        self.details = dict(type="all pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_allpass(self.fs, self.n_in, f, q)
        return self

    def make_peaking(self, f: float, q: float, boost_db: float) -> "Biquad":
        """
        Make this biquad a peaking.
        """
        self.details = dict(type="peaking", **_ws(locals()))
        self.dsp_block =  bq.biquad_peaking(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_constant_q(self, f: float, q: float, boost_db: float) -> "Biquad":
        """
        Make this biquad a constant q.
        """
        self.details = dict(type="constant q", **_ws(locals()))
        self.dsp_block =  bq.biquad_constant_q(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_lowshelf(self, f: float, q: float, boost_db: float) -> "Biquad":
        """
        Make this biquad a lowshelf.
        """
        self.details = dict(type="lowshelf", **_ws(locals()))
        self.dsp_block =  bq.biquad_lowshelf(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_highshelf(self, f: float, q: float, boost_db: float) -> "Biquad":
        """
        Make this biquad a highshelf.
        """
        self.details = dict(type="highshelf", **_ws(locals()))
        self.dsp_block =  bq.biquad_highshelf(self.fs, self.n_in, f, q, boost_db)
        return self

    def make_linkwitz(self, f0: float, q0: float, fp: float, qp: float) -> "Biquad":
        """
        Make this biquad a linkwitz.
        """
        self.details = dict(type="linkwitz", **_ws(locals()))
        self.dsp_block =  bq.biquad_linkwitz(self.fs, self.n_in, f0, q0, fp, qp)
        return self
