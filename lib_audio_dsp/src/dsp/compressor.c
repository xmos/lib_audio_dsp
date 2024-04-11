// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

#include <math.h>

compressor_t adsp_compressor_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  compressor_t comp;
  comp.env_det = adsp_env_detector_init(fs, attack_t, release_t, 0);
  float th = powf(10, threshold_db / 10);
  if (th > 1) th = 1.0;
  comp.threshold = from_float_pos(th);
  comp.gain = INT32_MAX;
  comp.slope = (1 - 1 / ratio) / 2;
  return comp;
}

static inline int32_t calc_rms_comp_gain(int32_t th, int32_t env, float slope) {
  // will calculate (th/env)^slope
  // assumes that th and env > 0 and slope [0, 1/2],
  // so that output fits in q31
  int32_t ah = 0, al = 0, r = 0, new_gain = 0;
  float ng_fl = 0;
  asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (th), "r" (Q_alpha), "0" (ah), "1" (al));
  asm("ldivu %0, %1, %2, %3, %4": "=r" (new_gain), "=r" (r): "r" (ah), "r" (al), "r" (env));
  r = -Q_alpha + 23;
  asm("fmake %0, %1, %2, %3, %4": "=r" (ng_fl): "r" (0), "r" (r), "r" (0), "r" (new_gain));
  ng_fl = powf(ng_fl, slope);
  asm("fsexp %0, %1, %2": "=r" (al), "=r" (r): "r" (ng_fl));
  asm("fmant %0, %1": "=r" (new_gain): "r" (ng_fl));
  r = -Q_alpha - r + 23;
  new_gain >>= r;
  return new_gain;
}

int32_t adsp_compressor_rms(
  compressor_t * comp,
  int32_t new_samp
) {
  adsp_env_detector_rms(&comp->env_det, new_samp);
  int32_t env = (comp->env_det.envelope == 0) ? 1 : comp->env_det.envelope;
  // this assumes that both th and env > 0 and that the slope is [0, 1/2]
  // basically (th/env)^slope > 1 is th^slope > env^slope
  // so if th and env both positive we can try to drop the slope
  // if slope == 0 expression fails, if slope is positive and th > env - passes
  int32_t new_gain = INT32_MAX;
  if ((comp->slope > 0) && (comp->threshold < env)) {
    new_gain = calc_rms_comp_gain(comp->threshold, env, comp->slope);
  }

  int32_t alpha = comp->env_det.release_alpha;
  if( comp->gain > new_gain ) {
    alpha = comp->env_det.attack_alpha;
  }

  comp->gain = q31_ema(comp->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, comp->gain);
}

int32_t adsp_compressor_rms_sidechain(
  compressor_t * comp,
  int32_t input_samp,
  int32_t detect_samp
) {
  adsp_env_detector_rms(&comp->env_det, detect_samp);
  int32_t env = (comp->env_det.envelope == 0) ? 1 : comp->env_det.envelope;
  // this assumes that both th and env > 0 and that the slope is [0, 1/2]
  // basically (th/env)^slope > 1 is th^slope > env^slope
  // so if th and env both positive we can try to drop the slope
  // if slope == 0 expression fails, if slope is positive and th > env - passes
  int32_t new_gain = INT32_MAX;
  if ((comp->slope > 0) && (comp->threshold < env)) {
    new_gain = calc_rms_comp_gain(comp->threshold, env, comp->slope);
  }

  int32_t alpha = comp->env_det.release_alpha;
  if( comp->gain > new_gain ) {
    alpha = comp->env_det.attack_alpha;
  }

  comp->gain = q31_ema(comp->gain, new_gain, alpha);
  return apply_gain_q31(input_samp, comp->gain);
}
