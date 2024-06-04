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
  float release_t
) {
  env_detector_t env_det;

  env_det.attack_alpha = get_alpha(fs, attack_t);
  env_det.release_alpha = get_alpha(fs, release_t);
  env_det.envelope = 0;

  return env_det;
}
