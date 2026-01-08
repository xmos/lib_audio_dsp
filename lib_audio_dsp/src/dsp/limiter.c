// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

int32_t adsp_limiter_peak(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_peak(&lim->env_det, new_samp);
  int32_t env = (lim->env_det.envelope == 0) ? 1 : lim->env_det.envelope;
  int32_t new_gain = INT32_MAX;
  if(lim->threshold < env) {
    int32_t ah = 0, al = 0, r = 0;
    asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (lim->threshold), "r" (31), "0" (ah), "1" (al));
    asm("ldivu %0, %1, %2, %3, %4": "=r" (new_gain), "=r" (r): "r" (ah), "r" (al), "r" (env));
  }

  int32_t alpha = lim->env_det.release_alpha;
  if( lim->gain > new_gain ) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = q31_ema(lim->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, lim->gain);
}

int32_t adsp_hard_limiter_peak(
  limiter_t * lim,
  int32_t new_samp
) {
  int32_t out = adsp_limiter_peak(lim, new_samp);
  // hard clip if above threshold
  out = (out > lim->threshold) ? lim->threshold : (out < -lim->threshold) ? -lim->threshold : out;
  return out;
}

int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_rms(&lim->env_det, new_samp);
  int32_t env = (lim->env_det.envelope == 0) ? 1 : lim->env_det.envelope;
  int32_t new_gain = INT32_MAX;
  if(lim->threshold < env) {
    int32_t ah = 0, al = 0; int r = 0;
    asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (lim->threshold), "r" (31), "0" (ah), "1" (al));
    asm("ldivu %0, %1, %2, %3, %4": "=r" (new_gain), "=r" (r): "r" (ah), "r" (al), "r" (env));
    new_gain = s32_sqrt(&r, new_gain, -31, S32_SQRT_MAX_DEPTH);
    right_shift_t rsh = -31 - r;
    new_gain >>= rsh;
  }

  float alpha = lim->env_det.release_alpha;
  if( lim->gain > new_gain ) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = q31_ema(lim->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, lim->gain);
  return new_samp;
}

int32_t adsp_clipper(
  clipper_t clip,
  int32_t new_samp
) {
  return (new_samp > clip) ? clip : (new_samp < -clip) ? -clip : new_samp;
}
