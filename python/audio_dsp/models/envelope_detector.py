# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Envelope detector Stages measure how the average or peak amplitude of
a signal varies over time.
"""

# from ..dsp import drc as drc
# from ..dsp import generic as dspg
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .stage import StageConfig, StageModel, StageParameters


class EnvelopeDetectorPlacement(BaseModel, extra="forbid"):
    input: list[int] = Field(
        default=[],
        description="Set of input edges, edges must be unique and not referenced anywhere else. Use the Fork stage to re-use edges.",
    )
    output: list[int] = Field([], max_length=0)
    name: str
    thread: int = Field(ge=0, lt=5)

    @field_validator("input", "output", mode="before")
    def _single_to_list(cls, value: int | list) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]


class EnvelopeDetectorParameters(StageParameters):
    attack_t: float = Field(default=0)
    release_t: float = Field(default=0)


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
