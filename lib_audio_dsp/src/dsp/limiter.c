// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

#include <math.h>


static inline int32_t from_float_pos(float val) {
  // assimes that val is positive
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (val));
  asm("fmant %0, %1": "=r" (mant): "r" (val));
  // mant to SIG_EXP
  right_shift_t shr = SIG_EXP - exp + 23;
  mant >>= shr;
  return mant;
}

limiter_t adsp_limiter_peak_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t
) {
  limiter_t lim;
  lim.env_det = adsp_env_detector_init(fs, atack_t, release_t, 0);
  float th = powf(10, threshold_db / 20);
  lim.threshold = from_float_pos(th);
  lim.gain = INT32_MAX;
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
  float th = powf(10, threshold_db / 10);
  lim.threshold = from_float_pos(th);
  lim.gain = INT32_MAX;
  return lim;
}

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
