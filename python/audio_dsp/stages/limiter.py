from ..design.stage import Stage, find_config
from ..dsp import drc as drc
from ..dsp import generic as dspg


class LimiterRMS(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_rms"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.limiter_rms(self.fs, self.n_in, threshold, at, rt)

        self.set_control_field_cb("attack_alpha",
                                  lambda: self.dsp_block.attack_alpha_f32)
        self.set_control_field_cb("release_alpha",
                                  lambda: self.dsp_block.release_alpha_f32)
        self.set_control_field_cb("threshold",
                                  lambda: self.dsp_block.threshold_f32)

    def make_limiter_rms(self, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        """
        Update limiter configuration based on new parameters.
        """
        self.details = dict(threshold_db=threshold_db, attack_t=attack_t, release_t=release_t, delay=delay, Q_sig=Q_sig)
        self.dsp_block = drc.limiter_rms(self.fs, self.n_in, threshold_db, attack_t, release_t, delay, Q_sig)
        return self

class LimiterPeak(Stage):
    def __init__(self, **kwargs):
        super().__init__(config=find_config("limiter_peak"), **kwargs)
        self.create_outputs(self.n_in)

        threshold = 0
        at = 0.01
        rt = 0.2
        self.dsp_block = drc.limiter_peak(self.fs, self.n_in, threshold, at, rt)

        self.set_control_field_cb("attack_alpha",
                                  lambda: self.dsp_block.attack_alpha_f32)
        self.set_control_field_cb("release_alpha",
                                  lambda: self.dsp_block.release_alpha_f32)
        self.set_control_field_cb("threshold",
                                  lambda: self.dsp_block.threshold_f32)

    def make_limiter_peak(self, threshold_db, attack_t, release_t, delay=0, Q_sig=dspg.Q_SIG):
        """
        Update limiter configuration based on new parameters.
        """
        self.details = dict(threshold_db=threshold_db, attack_t=attack_t, release_t=release_t, delay=delay, Q_sig=Q_sig)
        self.dsp_block = drc.limiter_peak(self.fs, self.n_in, threshold_db, attack_t, release_t, delay, Q_sig)
        return self
