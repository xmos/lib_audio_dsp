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
  return apply_gain(new_samp, lim->gain);
}

int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_rms(&lim->env_det, new_samp);
  float env = (lim->env_det.envelope == 0) ? 1e-20 : lim->env_det.envelope;
  float new_gain = (lim->threshold > env) ? 1 : sqrtf(lim->threshold / env);

  float alpha = lim->env_det.release_alpha;
  if( lim->gain > new_gain ) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = lim->gain + alpha * (new_gain - lim->gain);
  return apply_gain(new_samp, lim->gain);
}
