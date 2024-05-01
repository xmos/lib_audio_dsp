// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

void adsp_expander_set_th(
  expander_t * ex,
  int32_t new_th
) {
  // Avoid division by zero
  ex->threshold = (!new_th) ? 1 : new_th;
  // x * 2 ^ -63 / y * 2 ^ -27 = xy * 2 ^ -36
  ex->inv_threshold =  INT64_MAX / ex->threshold;
}

expander_t adsp_expander_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  expander_t ex;
  ex.env_det = adsp_env_detector_init(fs, attack_t, release_t);
  float th = powf(10, threshold_db / 20);
  adsp_expander_set_th(&ex, from_float_pos(th));
  ex.gain = INT32_MAX;
  ex.slope = 1 - ratio;
  ex.env_det.envelope = 1 << (-SIG_EXP);
  return ex;
}

int32_t adsp_expander(
  expander_t * ex,
  int32_t new_samp
) {
  adsp_env_detector_peak(&ex->env_det, new_samp);
  int32_t new_gain = INT32_MAX;

  if (-ex->slope > 0 && ex->threshold > ex->env_det.envelope) {
    // This looks a bit scary, but as long as envelope < threshold,
    // it can't overflow. The inv_th had exp of -36, when multiplied by env
    // has exp of -63.
    int64_t new_gain_i64 =  ex->env_det.envelope * ex->inv_threshold;
    new_gain = new_gain_i64 >> 32;
    int32_t exp = -Q_alpha - 32 + 23;
    float ng_fl;
    asm("fmake %0, %1, %2, %3, %4": "=r" (ng_fl): "r" (0), "r" (exp), "r" (new_gain), "r" ((uint32_t)new_gain_i64));
    ng_fl = powf(ng_fl, -ex->slope);
    asm("fsexp %0, %1, %2": "=r" (new_gain), "=r" (exp): "r" (ng_fl));
    asm("fmant %0, %1": "=r" (new_gain): "r" (ng_fl));
    exp = -Q_alpha - exp + 23;
    new_gain >>= exp;
  }
  int32_t alpha = ex->env_det.attack_alpha;
  if( ex->gain > new_gain ) {
    alpha = ex->env_det.release_alpha;
  }

  ex->gain = q31_ema(ex->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, ex->gain);
}
