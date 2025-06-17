# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""The pydantic models of the DSP Stages."""

from .signal_chain import VolumeControl, FixedGain, Fork, Mixer, Delay

from .cascaded_biquads import ParametricEq8b, ParametricEq16b, NthOrderFilter
from .reverb import ReverbPlateStereo
from .envelope_detector import EnvelopeDetectorPeak, EnvelopeDetectorRMS
from .noise_suppressor_expander import NoiseSuppressorExpander
from .biquad import Biquad
from .limiter import LimiterRMS, LimiterPeak, HardLimiterPeak
from .noise_gate import NoiseGate
from .compressor import CompressorRMS
from .compressor_sidechain import CompressorSidechain
from .fir import FirDirect
from .graphic_eq import GraphicEq10b
