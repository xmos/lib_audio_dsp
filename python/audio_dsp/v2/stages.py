from typing import Literal
from .types import DspStage, TuningBaseModel, ConstantsBaseModel, UnknownEdge, stage_helpers

reverb = DspStage()


@reverb.tuning
class ReverbTuningFields(TuningBaseModel):
    predelay: int = 0
    damping: float = 0.0
    decay: float = 0.0


@reverb.constants
class ReverbConstantsFields(ConstantsBaseModel):
    max_room_size: float = 1.0
    max_predelay: float = 20.0


biquad = DspStage()


@biquad.tuning
class BiquadTuning(TuningBaseModel):
    type: Literal["lowpass", "highpass", "allpass"]
    q: float
    f: float


@reverb.determine_outputs
@biquad.determine_outputs
def _determine_outputs(context, inputs):
    # TODO - This is too naive. When the user drops a stage into a GUI they will expect to
    #        see what its input types are without trying to make connections and seeing if it
    #        fails. Also stages should have named inputs and outputs. For example in an n-channel switch
    #        it should be clear which inputs map to each switch position. In a side chain compressor
    #        it should be clear which is the input, and which is the detect.
    # test inputs
    return stage_helpers.outputs_match_inputs(context, inputs, len(inputs), ["int"])


mixer = DspStage()


@mixer.determine_outputs
def mixer_determine_outputs(context, inputs):
    # test inputs
    return stage_helpers.outputs_match_inputs(context, inputs, 1, ["int"])
