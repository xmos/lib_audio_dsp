// Copyright 2024 XMOS ngITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"

static inline int32_t apply_gain_q31(int32_t samp, q1_31 gain) {
  int32_t q = 31;
  int32_t ah = 0, al = 1 << (q - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (samp), "r" (gain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));

  return ah;
}

static inline int32_t q31_ema(int32_t x, int32_t samp, q1_31 alpha) {
  // this assumes that x and samp are positive and alpha is q31
  // x and samp have to have the same exponent
  int32_t ah, al;
  int32_t mul = samp - x;

  // preload the acc with x at position of 31 
  // (essentially giving it exponent of -31 + x.exp)
  asm("linsert %0, %1, %2, %3, 32":"=r" (ah), "=r" (al): "r"(x), "r"(31), "0"(0), "1" (0));
  // x + alpha * (samp - x) with exponent -31 + x.exp
  asm("maccs %0,%1,%2,%3":"=r"(ah),"=r"(al):"r"(alpha),"r"(mul), "0" (ah), "1" (al));
  // saturate and extract from 63rd bit
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (31), "0" (ah), "1" (al));
  asm("lextract %0,%1,%2,%3,32":"=r"(x):"r"(ah),"r"(al),"r"(31));
  return x;
}

noise_gate_t adsp_noise_gate_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t
) {
  return (noise_gate_t)adsp_limiter_peak_init(fs, threshold_db, atack_t, release_t);
}

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
