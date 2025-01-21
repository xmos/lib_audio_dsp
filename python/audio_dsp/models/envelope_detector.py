# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Envelope detector Stages measure how the average or peak amplitude of
a signal varies over time.
"""

from .stage import StageModel, StageParameters, StageConfig
# from ..dsp import drc as drc
# from ..dsp import generic as dspg
from typing import Literal

from pydantic import Field


class EnvelopeDetectorParameters(StageParameters):
    attack_t: float = Field(default=0)
    release_t: float = Field(default=0)


class EnvelopeDetectorPeak(StageModel):
    """
    A stage with no outputs that measures the signal peak envelope.

    The current envelope of the signal can be read out using this stage's
    ``envelope`` control.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.envelope_detector_peak`
        The DSP block class; see :ref:`EnvelopeDetectorPeak`
        for implementation details.

    """

    op_type: Literal["EnvelopeDetectorPeak"] = "EnvelopeDetectorPeak"
    parameters: EnvelopeDetectorParameters = Field(default_factory=EnvelopeDetectorParameters)


class EnvelopeDetectorRMS(StageModel):
    """
    A stage with no outputs that measures the signal RMS envelope.

    The current envelope of the signal can be read out using this stage's
    ``envelope`` control.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.drc.envelope_detector_rms`
        The DSP block class; see :ref:`EnvelopeDetectorRMS`
        for implementation details.

    """

    op_type: Literal["EnvelopeDetectorRms"] = "EnvelopeDetectorRms"
    parameters: EnvelopeDetectorParameters = Field(default_factory=EnvelopeDetectorParameters)
