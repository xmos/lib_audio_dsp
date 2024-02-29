// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include <math.h>

static inline int32_t apply_gain(int32_t samp, float gain) {
  int32_t mant, exp, zero;
  asm("fsexp %0, %1, %2": "=r" (zero), "=r" (exp): "r" (gain));
  asm("fmant %0, %1": "=r" (mant): "r" (gain));

  int32_t q = -(exp - 23);
  int32_t ah = 0, al = 1 << (q - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (samp), "r" (mant), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));

  return ah;
}

compressor_t adsp_compressor_rms_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t,
  float ratio
) {
  compressor_t comp;
  comp.env_det = adsp_env_detector_init(fs, atack_t, release_t, 0);
  comp.threshold = powf(10, threshold_db / 10);
  comp.gain = 1;
  comp.slope = (1 - 1 / ratio) / 2;
  return comp;
}

int32_t adsp_compressor_rms(
  compressor_t * comp,
  int32_t new_samp
) {
  adsp_env_detector_rms(&comp->env_det, new_samp);
  float env = comp->env_det.envelope;
  float th = comp->threshold;
  env = (env == 0) ? 1e-20 : env;
  // this assumes that both th and env > 0 and that the slope is [0, 1/2]
  // basically (th/env)^slope > 1 is th^slope > env^slope
  // so if th and env both positive we can try to drop the slope
  // if slope == 0 expression fails, if slope is positive and th > env - passes
  float new_gain = ((comp->slope > 0) && (th < env)) ? powf((th / env), comp->slope) : 1;

  float alpha = comp->env_det.release_alpha;
  if( comp->gain > new_gain ) {
    alpha = comp->env_det.attack_alpha;
  }

  comp->gain = comp->gain + alpha * (new_gain - comp->gain);
  return apply_gain(new_samp, comp->gain);
}
