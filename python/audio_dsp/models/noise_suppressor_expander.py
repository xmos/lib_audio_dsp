# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Noise suppressor and expander Stages control the behaviour
of quiet signals, typically by tring to reduce the audibility of noise
in the signal.
"""

from .stage import StageModel, StageParameters, StageConfig
# from ..dsp import drc as drc
# from ..dsp import generic as dspg

from pydantic import Field
from typing import Literal


class NoiseSuppressorExpanderParameters(StageParameters):
    ratio: float = Field(default=3)
    threshold_db: float = Field(default=-35)
    attack_t: float = Field(default=0.005)
    release_t: float = Field(default=0.120)


class NoiseSuppressorExpander(StageModel):
    """The Noise Suppressor (Expander) stage. A noise suppressor that
    reduces the level of an audio signal when it falls below a
    threshold. This is also known as an expander.

    When the signal envelope falls below the threshold, the gain applied
    to the signal is reduced relative to the expansion ratio over the
    release time. When the envelope returns above the threshold, the
    gain applied to the signal is increased to 1 over the attack time.

    The initial state of the noise suppressor is with the suppression
    off; this models a full scale signal having been present before
    t = 0.

    Attributes
    ----------
    dsp_block : :class:`audio_dsp.dsp.drc.expander.noise_suppressor_expander`
        The DSP block class; see :ref:`NoiseSuppressorExpander`
        for implementation details.
    """

    op_type: Literal["NoiseSuppressorExpander"] = "NoiseSuppressorExpander"
    parameters: NoiseSuppressorExpanderParameters = Field(
        default_factory=NoiseSuppressorExpanderParameters
    )
