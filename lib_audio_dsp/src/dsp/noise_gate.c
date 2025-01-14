// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

int32_t adsp_noise_gate(
  noise_gate_t * ng,
  int32_t new_samp
) {
  adsp_env_detector_peak(&ng->env_det, new_samp);
  int32_t env = (ng->env_det.envelope == 0) ? 1 : ng->env_det.envelope;
  int32_t new_gain = (ng->threshold > env) ? 0 : INT32_MAX;

  // for the noise gate, the attack and release times are swapped
  // i.e. attack time is after going under threshold instead of over
  int32_t alpha = ng->env_det.attack_alpha;
  if( ng->gain > new_gain ) {
    alpha = ng->env_det.release_alpha;
  }

  ng->gain = q31_ema(ng->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, ng->gain);
}
