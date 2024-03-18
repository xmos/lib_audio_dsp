// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include <math.h>

static inline int32_t apply_gain_q31(int32_t samp, q1_31 gain) {
  int32_t q = 31;
  int32_t ah = 0, al = 1 << (q - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (samp), "r" (gain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));

  return ah;
}

static inline int32_t weird_ema(int32_t samp, int32_t env, int32_t alpha) {
  // this assumes that env and samp are positive and alpha is q31
  int32_t ah, al;
  int32_t mul = samp - env;

  // preload the acc with env at position of 31 
  // (essentially giving it exponent of -58, rather then -27)
  asm("linsert %0, %1, %2, %3, 32":"=r" (ah), "=r" (al): "r"(env), "r"(31), "0"(0), "1" (0));
  // env + alpha * (samp - env) with exponent -58
  asm("maccs %0,%1,%2,%3":"=r"(ah),"=r"(al):"r"(alpha),"r"(mul), "0" (ah), "1" (al));
  // saturate and extract from 63rd bit
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (31), "0" (ah), "1" (al));
  asm("lextract %0,%1,%2,%3,32":"=r"(env):"r"(ah),"r"(al),"r"(31));
  return env;
}

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

compressor_t adsp_compressor_rms_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t,
  float ratio
) {
  compressor_t comp;
  comp.env_det = adsp_env_detector_init(fs, atack_t, release_t, 0);
  float th = powf(10, threshold_db / 10);
  comp.threshold = from_float_pos(th);
  comp.gain = INT32_MAX;
  comp.slope = (1 - 1 / ratio) / 2;
  return comp;
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
    int32_t ah = 0, al = 0, r = 0;
    float ng_fl = 0;
    asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (comp->threshold), "r" (31), "0" (ah), "1" (al));
    asm("ldivu %0, %1, %2, %3, %4": "=r" (new_gain), "=r" (r): "r" (ah), "r" (al), "r" (env));
    r = -31 + 23;
    asm("fmake %0, %1, %2, %3, %4": "=r" (ng_fl): "r" (0), "r" (r), "r" (0), "r" (new_gain));
    ng_fl = powf(ng_fl, comp->slope);
    asm("fsexp %0, %1, %2": "=r" (al), "=r" (r): "r" (ng_fl));
    asm("fmant %0, %1": "=r" (new_gain): "r" (ng_fl));
    r = -31 - r + 23;
    new_gain >>= r;
  }

  int32_t alpha = comp->env_det.release_alpha;
  if( comp->gain > new_gain ) {
    alpha = comp->env_det.attack_alpha;
  }

  comp->gain = weird_ema(new_gain, comp->gain, alpha);
  return apply_gain_q31(new_samp, comp->gain);
}
