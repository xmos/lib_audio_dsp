
#include "dsp/adsp.h"

#include <xcore/assert.h>

#define Q_sig -27

static inline int32_t float_to_fixed(float x, exponent_t output_exp){
  float_s32_t v = f32_to_float_s32(x);
  right_shift_t shr = output_exp - v.exp;
  if(shr >= 0) return (v.mant >> ( shr) );
  else         return (v.mant << (-shr) );
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

  fs = 2 / fs;
  env_det.attack_alpha = float_to_fixed(fs / attack_t, -30);
  env_det.release_alpha = float_to_fixed(fs / release_t, -30);
  env_det.envelope = (float_s32_t){0, Q_sig};

  return env_det;
}

void adsp_env_detector_peak(
  env_detector_t * env_det,
  int32_t new_sample
) {
  float_s32_t samp = (float_s32_t){new_sample, Q_sig};
  samp = float_s32_abs(samp);

  uq2_30 alpha = env_det->release_alpha;
  if (float_s32_gt(samp, env_det->envelope)) {
    alpha = env_det->attack_alpha;
  }

  env_det->envelope = float_s32_ema(samp, env_det->envelope, alpha);
}

void adsp_env_detector_rms(
  env_detector_t * env_det,
  int32_t new_sample
) {
  float_s32_t samp = (float_s32_t){new_sample, Q_sig};
  samp = float_s32_mul(samp, samp);

  uq2_30 alpha = env_det->release_alpha;
  if (float_s32_gt(samp, env_det->envelope)) {
    alpha = env_det->attack_alpha;
  }

  env_det->envelope = float_s32_ema(samp, env_det->envelope, alpha);
}
