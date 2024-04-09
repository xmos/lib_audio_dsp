// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

void adsp_noise_suppressor_set_th(
  noise_suppressor_t * ns,
  int32_t new_th
) {
  // Avoid division by zero
  ns->threshold = (!new_th) ? 1 : new_th;
  // x * 2 ^ -63 / y * 2 ^ -27 = xy * 2 ^ -36
  ns->inv_threshold =  INT64_MAX / ns->threshold;
}

noise_suppressor_t adsp_noise_suppressor_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  noise_suppressor_t ns;
  ns.env_det = adsp_env_detector_init(fs, attack_t, release_t, 0);
  float th = powf(10, threshold_db / 20);
  adsp_noise_suppressor_set_th(&ns, from_float_pos(th));
  ns.gain = INT32_MAX;
  ns.slope = 1 - ratio;
  ns.env_det.envelope = 1 << (-SIG_EXP);
  return ns;
}

int32_t adsp_noise_suppressor(
  noise_suppressor_t * ns,
  int32_t new_samp
) {
  adsp_env_detector_peak(&ns->env_det, new_samp);
  int32_t new_gain = INT32_MAX;

  if (-ns->slope > 0 && ns->threshold > ns->env_det.envelope) {
    // This looks a bit scary, but as long as envelope < threshold,
    // it can't overflow. The inv_th had exp of -36, when multiplied by env
    // has exp of -63.
    int64_t new_gain_i64 =  ns->env_det.envelope * ns->inv_threshold;
    new_gain = new_gain_i64 >> 32;
    int32_t exp = -Q_alpha - 32 + 23;
    float ng_fl;
    asm("fmake %0, %1, %2, %3, %4": "=r" (ng_fl): "r" (0), "r" (exp), "r" (new_gain), "r" ((uint32_t)new_gain_i64));
    ng_fl = powf(ng_fl, -ns->slope);
    asm("fsexp %0, %1, %2": "=r" (new_gain), "=r" (exp): "r" (ng_fl));
    asm("fmant %0, %1": "=r" (new_gain): "r" (ng_fl));
    exp = -Q_alpha - exp + 23;
    new_gain >>= exp;
  }
  int32_t alpha = ns->env_det.attack_alpha;
  if( ns->gain > new_gain ) {
    alpha = ns->env_det.release_alpha;
  }

  ns->gain = q31_ema(ns->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, ns->gain);
}
