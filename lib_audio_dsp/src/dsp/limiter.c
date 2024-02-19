// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"

static inline int32_t f32_to_fixed(float x, exponent_t output_exp){
  float_s32_t v = f32_to_float_s32(x);
  right_shift_t shr = output_exp - v.exp;
  asm("ashr %0, %1, %2" : "=r" (v.mant) : "r" (v.mant), "r" (shr));
  return v.mant;
}

limiter_t adsp_limiter_peak_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t
) {
  limiter_t lim;
  lim.env_det = adsp_env_detector_init(fs, atack_t, release_t, 0);
  lim.threshold = powf(10, threshold_db / 20);
  lim.gain = 1;
  return lim;
}

limiter_t adsp_limiter_rms_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t
) {
  limiter_t lim;
  lim.env_det = adsp_env_detector_init(fs, atack_t, release_t, 0);
  lim.threshold = powf(10, threshold_db / 10);
  lim.gain = 1;
  return lim;
}

int32_t adsp_limiter_peak(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_peak(&lim->env_det, new_samp);
  float env = (lim->env_det.envelope == 0) ? 1e-20 : lim->env_det.envelope;
  float new_gain = (lim->threshold > env) ? 1 : lim->threshold / env;

  float alpha = lim->env_det.release_alpha;
  if( lim->gain > new_gain ) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = lim->gain + alpha * (new_gain - lim->gain);
  float y = float_s32_to_float((float_s32_t){new_samp, SIG_EXP});
  return f32_to_fixed(y * lim->gain, SIG_EXP);
}

int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_rms(&lim->env_det, new_samp);
  float env = (lim->env_det.envelope == 0) ? 1e-20 : lim->env_det.envelope;
  float new_gain = (lim->threshold > env) ? 1 : lim->threshold / env;

  float alpha = lim->env_det.release_alpha;
  if( lim->gain > new_gain ) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = lim->gain + alpha * (new_gain - lim->gain);
  float y = float_s32_to_float((float_s32_t){new_samp, SIG_EXP});
  return f32_to_fixed(y * lim->gain, SIG_EXP);
}
