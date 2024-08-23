// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "control/adsp_control.h"
#include "dsp/_helpers/drc_utils.h"

#include <xcore/assert.h>

#include <math.h>

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

limiter_t adsp_limiter_peak_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t
) {
  limiter_t lim;
  lim.env_det = adsp_env_detector_init(fs, attack_t, release_t);
  float th = powf(10, threshold_db / 20);
  lim.threshold = from_float_pos(th);
  lim.gain = INT32_MAX;
  return lim;
}

limiter_t adsp_limiter_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t
) {
  limiter_t lim;
  lim.env_det = adsp_env_detector_init(fs, attack_t, release_t);
  float th = powf(10, threshold_db / 10);
  lim.threshold = from_float_pos(th);
  lim.gain = INT32_MAX;
  return lim;
}

compressor_t adsp_compressor_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  compressor_t comp;
  comp.env_det = adsp_env_detector_init(fs, attack_t, release_t);
  float th = powf(10, threshold_db / 10);
  if (th > 1) th = 1.0;
  comp.threshold = from_float_pos(th);
  comp.gain = INT32_MAX;
  comp.slope = (1 - 1 / ratio) / 2;
  return comp;
}

noise_gate_t adsp_noise_gate_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t
) {
  noise_gate_t ng = (noise_gate_t)adsp_limiter_peak_init(fs, threshold_db, attack_t, release_t);
  ng.env_det.envelope = (1 << (-SIG_EXP)) - 1;
  return ng;
}

void adsp_noise_suppressor_expander_set_th(
  noise_suppressor_expander_t * nse,
  int32_t new_th
) {
  // Avoid division by zero
  nse->threshold = (!new_th) ? 1 : new_th;
  // x * 2 ^ -63 / y * 2 ^ -27 = xy * 2 ^ -36
  nse->inv_threshold =  INT64_MAX / nse->threshold;
}

noise_suppressor_expander_t adsp_noise_suppressor_expander_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  noise_suppressor_expander_t nse;
  nse.env_det = adsp_env_detector_init(fs, attack_t, release_t);
  float th = powf(10, threshold_db / 20);
  adsp_noise_suppressor_expander_set_th(&nse, from_float_pos(th));
  nse.gain = INT32_MAX;
  nse.slope = 1 - ratio;
  nse.env_det.envelope = (1 << (-SIG_EXP)) - 1;
  return nse;
}
