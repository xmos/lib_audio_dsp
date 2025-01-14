// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

int32_t adsp_noise_suppressor_expander(
  noise_suppressor_expander_t * nse,
  int32_t new_samp
) {
  adsp_env_detector_peak(&nse->env_det, new_samp);
  int32_t new_gain = INT32_MAX;

  if (-nse->slope > 0 && nse->threshold > nse->env_det.envelope) {
    // This looks a bit scary, but as long as envelope < threshold,
    // it can't overflow. The inv_th had exp of -36, when multiplied by env
    // has exp of -63.
    int64_t new_gain_i64 =  nse->env_det.envelope * nse->inv_threshold;
    new_gain = new_gain_i64 >> 32;
    int32_t exp = -Q_alpha - 32 + 23;
    float ng_fl;
    asm("fmake %0, %1, %2, %3, %4": "=r" (ng_fl): "r" (0), "r" (exp), "r" (new_gain), "r" ((uint32_t)new_gain_i64));
    ng_fl = powf(ng_fl, -nse->slope);
    asm("fsexp %0, %1, %2": "=r" (new_gain), "=r" (exp): "r" (ng_fl));
    asm("fmant %0, %1": "=r" (new_gain): "r" (ng_fl));
    exp = -Q_alpha - exp + 23;
    new_gain >>= exp;
  }
  int32_t alpha = nse->env_det.attack_alpha;
  if( nse->gain > new_gain ) {
    alpha = nse->env_det.release_alpha;
  }

  nse->gain = q31_ema(nse->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, nse->gain);
}
