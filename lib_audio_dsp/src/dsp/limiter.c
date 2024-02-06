
#include "dsp/adsp.h"

static const float_s32_t one = (float_s32_t){0x40000000, -30};
static const float_s32_t delta = (float_s32_t){1, -60};

static inline int32_t float_s32_to_fixed(float_s32_t v, exponent_t output_exp){
  right_shift_t shr = output_exp - v.exp;
  if(shr >= 0) return (v.mant >> ( shr ));
  else         return (v.mant << (-shr ));
}

limiter_t adsp_limiter_peak_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t
) {
  limiter_t lim;
  lim.env_det = adsp_env_detector_init(fs, atack_t, release_t, 0);
  lim.threshold = f32_to_float_s32(powf(10, threshold_db / 20));
  lim.gain = one;
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
  lim.threshold = f32_to_float_s32(powf(10, threshold_db / 10));
  lim.gain = one;
  return lim;
}

int32_t adsp_limiter_peak(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_peak(&lim->env_det, new_samp);
  float_s32_t env = (lim->env_det.envelope.mant == 0) ? delta : lim->env_det.envelope;
  float_s32_t new_gain = (float_s32_gt(lim->threshold, env)) ? one : float_s32_div(lim->threshold, env);

  uq2_30 alpha = lim->env_det.release_alpha;
  if (float_s32_gt(lim->gain, new_gain)) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = float_s32_ema(new_gain, lim->gain, alpha);
  float_s32_t y = float_s32_mul((float_s32_t){new_samp, SIG_EXP}, lim->gain);
  return float_s32_to_fixed(y, SIG_EXP);
}

int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp
) {
  adsp_env_detector_rms(&lim->env_det, new_samp);
  float_s32_t env = (lim->env_det.envelope.mant == 0) ? delta : lim->env_det.envelope;
  float_s32_t new_gain = (float_s32_gt(lim->threshold, env)) ? one : float_s32_div(lim->threshold, env);

  uq2_30 alpha = lim->env_det.release_alpha;
  if (float_s32_gt(lim->gain, new_gain)) {
    alpha = lim->env_det.attack_alpha;
  }

  lim->gain = float_s32_ema(new_gain, lim->gain, alpha);
  float_s32_t y = float_s32_mul((float_s32_t){new_samp, SIG_EXP}, lim->gain);
  return float_s32_to_fixed(y, SIG_EXP);
}
