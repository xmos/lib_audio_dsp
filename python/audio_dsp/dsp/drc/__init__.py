# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""This sub-package contains all the dynamic range control (DRC)
DSP components.
"""

from audio_dsp.dsp.drc.drc import (
    compressor_limiter_base as compressor_limiter_base,
    envelope_detector_peak as envelope_detector_peak,
    envelope_detector_rms as envelope_detector_rms,
    limiter_peak as limiter_peak,
    hard_limiter_peak as hard_limiter_peak,
    limiter_rms as limiter_rms,
    compressor_rms as compressor_rms,
    compressor_rms_softknee as compressor_rms_softknee,
    clipper as clipper,
)
from audio_dsp.dsp.drc.stereo_compressor_limiter import (
    limiter_peak_stereo as limiter_peak_stereo,
    compressor_rms_stereo as compressor_rms_stereo,
)

from audio_dsp.dsp.drc.sidechain import (
    compressor_rms_sidechain_mono as compressor_rms_sidechain_mono,
    compressor_rms_sidechain_stereo as compressor_rms_sidechain_stereo,
)

from audio_dsp.dsp.drc.expander import (
    noise_gate as noise_gate,
    noise_suppressor_expander as noise_suppressor_expander,
)
