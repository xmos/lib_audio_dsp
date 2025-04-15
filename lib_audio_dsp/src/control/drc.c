// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "control/adsp_control.h"

env_detector_t adsp_env_detector_init(
  float fs,
  float attack_t,
  float release_t
) {
  env_detector_t env_det;
  env_det.attack_alpha = calc_alpha(fs, attack_t);
  env_det.release_alpha = calc_alpha(fs, release_t);
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
  lim.threshold = db_to_q_sig(threshold_db);
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
  lim.threshold = db_pow_to_q_sig(threshold_db);
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
  comp.threshold = db_pow_to_q_sig(threshold_db);
  comp.gain = INT32_MAX;
  comp.slope = (1.0f - 1.0f / ratio) / 2.0f;
  return comp;
}

compressor_stereo_t adsp_compressor_rms_stereo_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  compressor_stereo_t comp;
  comp.env_det_l = adsp_env_detector_init(fs, attack_t, release_t);
  comp.env_det_r = adsp_env_detector_init(fs, attack_t, release_t);
  comp.threshold = db_pow_to_q_sig(threshold_db);
  comp.gain = INT32_MAX;
  comp.slope = (1.0f - 1.0f / ratio) / 2.0f;
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
  int32_t th = db_to_q_sig(threshold_db);
  adsp_noise_suppressor_expander_set_th(&nse, th);
  nse.gain = INT32_MAX;
  nse.slope = 1 - ratio;
  nse.env_det.envelope = (1 << (-SIG_EXP)) - 1;
  return nse;
}
