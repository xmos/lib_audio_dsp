// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

#include <xcore/assert.h>

static inline q1_31 get_alpha(float fs, float time) {
  xassert(time > 0 && "time has to be positive");
  time = 2 / (fs * time);
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (time));
  asm("fmant %0, %1": "=r" (mant): "r" (time));

  // mant to q31
  right_shift_t shr = -Q_alpha - exp + 23;
  mant >>= shr;
  return mant;
}

env_detector_t adsp_env_detector_init(
  float fs,
  float attack_t,
  float release_t,
  float detect_t
) {
  env_detector_t env_det;

  if (detect_t && (attack_t || release_t)) {
    xassert(0 && "either detect_t OR (attack_t AND release_t) must be specified");
  } else if (detect_t) {
    attack_t = detect_t;
    release_t = detect_t;
  }

  env_det.attack_alpha = get_alpha(fs, attack_t);
  env_det.release_alpha = get_alpha(fs, release_t);
  env_det.envelope = 0;

  return env_det;
}

void adsp_env_detector_peak(
  env_detector_t * env_det,
  int32_t new_sample
) {
  if (new_sample < 0) {
    new_sample = - new_sample;
  }

  int32_t alpha = env_det->release_alpha;
  if (new_sample > env_det->envelope) {
    alpha = env_det->attack_alpha;
  }

  env_det->envelope = q31_ema(env_det->envelope, new_sample, alpha);
}

void adsp_env_detector_rms(
  env_detector_t * env_det,
  int32_t new_sample
) {
  int32_t ah, al;
  asm("maccs %0,%1,%2,%3":"=r"(ah),"=r"(al):"r"(new_sample),"r"(new_sample), "0" (0), "1" (1 << (Q_SIG - 1)));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (Q_SIG), "0" (ah), "1" (al));
  asm("lextract %0,%1,%2,%3,32":"=r"(new_sample):"r"(ah),"r"(al),"r"(Q_SIG));

  int32_t alpha = env_det->release_alpha;
  if (new_sample > env_det->envelope) {
    alpha = env_det->attack_alpha;
  }

  env_det->envelope = q31_ema(env_det->envelope, new_sample, alpha);
}
