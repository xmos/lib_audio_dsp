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
        self.create_outputs(self.n_in)
        self.set_control_field_cb("filter_coeffs",
                                  lambda: [i for i in self.get_fixed_point_coeffs()])
        self.set_control_field_cb("left_shift",
                                  lambda: self.dsp_block.b_shift)
        self.make_bypass()

    def get_fixed_point_coeffs(self):
        a = np.array(self.dsp_block.coeffs)
        return np.array(a*(2**30), dtype=np.int32)

    def make_bypass(self):
        self.details = {}
        self.dsp_block =  bq.biquad_bypass(self.fs)
        return self

    def make_lowpass(self, f, q):
        self.details = dict(type="low pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_lowpass(self.fs, f, q)
        return self

    def make_highpass(self, f, q):
        self.details = dict(type="high pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_highpass(self.fs, f, q)
        return self

    def make_bandpass(self, f, bw):
        self.details = dict(type="band pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_bandpass(self.fs, f, bw)
        return self

    def make_bandstop(self, f, bw):
        self.details = dict(type="band stop", **_ws(locals()))
        self.dsp_block =  bq.biquad_bandstop(self.fs, f, bw)
        return self

    def make_notch(self, f, q):
        self.details = dict(type="notch", **_ws(locals()))
        self.dsp_block =  bq.biquad_notch(self.fs, f, q)
        return self

    def make_allpass(self, f, q):
        self.details = dict(type="all pass", **_ws(locals()))
        self.dsp_block =  bq.biquad_allpass(self.fs, f, q)
        return self

    def make_peaking(self, f, q, boost_db):
        self.details = dict(type="peaking", **_ws(locals()))
        self.dsp_block =  bq.biquad_peaking(self.fs, f, q, boost_db)
        return self

    def make_constant_q(self, f, q, boost_db):
        self.details = dict(type="constant q", **_ws(locals()))
        self.dsp_block =  bq.biquad_constant_q(self.fs, f, q, boost_db)
        return self

    def make_lowshelf(self, f, q, boost_db):
        self.details = dict(type="lowshelf", **_ws(locals()))
        self.dsp_block =  bq.biquad_lowshelf(self.fs, f, q, boost_db)
        return self

    def make_highshelf(self, f, q, boost_db):
        self.details = dict(type="highshelf", **_ws(locals()))
        self.dsp_block =  bq.biquad_highshelf(self.fs, f, q, boost_db)
        return self

    def make_linkwitz(self, f0, q0, fp, qp):
        self.details = dict(type="linkwitz", **_ws(locals()))
        self.dsp_block =  bq.biquad_linkwitz(self.fs, f0, q0, fp, qp)
        return self
