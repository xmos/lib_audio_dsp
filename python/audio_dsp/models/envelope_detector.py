# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Envelope detector Stages measure how the average or peak amplitude of
a signal varies over time.
"""

# from ..dsp import drc as drc
# from ..dsp import generic as dspg
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from audio_dsp.models.stage import StageConfig, StageModel, StageParameters, NodePlacement
from audio_dsp.models.fields import DEFAULT_ATTACK_T, DEFAULT_RELEASE_T

class EnvelopeDetectorPlacement(NodePlacement):
    """Graph placement for an Envelope Stage. This stage has no outputs."""

    input: list[int] = Field(
        default=[],
        description="Set of input edges.",
    )
    output: list[int] = Field(default=[], max_length=0)


class EnvelopeDetectorParameters(StageParameters):
    """Parameters for an EnvelopeDetector Stage."""

    attack_t: float = DEFAULT_ATTACK_T()
    release_t: float = DEFAULT_RELEASE_T()


class EnvelopeDetectorPeak(StageModel[EnvelopeDetectorPlacement]):
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


class EnvelopeDetectorRMS(StageModel[EnvelopeDetectorPlacement]):
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

    op_type: Literal["EnvelopeDetectorRMS"] = "EnvelopeDetectorRMS"
    parameters: EnvelopeDetectorParameters = Field(default_factory=EnvelopeDetectorParameters)
