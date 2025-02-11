# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Collection of all XMOS DSP stages for use in DSP pipeline."""

from .biquad import Biquad, BiquadSlew
from .cascaded_biquads import CascadedBiquads, ParametricEq
from .limiter import LimiterRMS, LimiterPeak, HardLimiterPeak, Clipper
from .noise_gate import NoiseGate
from .noise_suppressor_expander import NoiseSuppressorExpander
from .signal_chain import (
    VolumeControl,
    FixedGain,
    Fork,
    Adder,
    Switch,
    SwitchStereo,
    Subtractor,
    Bypass,
    Mixer,
)
from .compressor import CompressorRMS
from .reverb import ReverbRoom, ReverbRoomStereo, ReverbPlateStereo
from .fir import FirDirect
from .compressor import CompressorRMS
from .compressor_sidechain import CompressorSidechain
from .envelope_detector import EnvelopeDetectorPeak, EnvelopeDetectorRMS
from .delay import DelayStage as Delay  # Replace signal_chain.Delay with our model-based one

# helper from design which allows listing all the available stages.
from ..design.stage import all_stages
