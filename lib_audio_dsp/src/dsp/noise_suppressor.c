// Copyright 2024 XMOS ngITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"

static inline int32_t from_float_pos(float val) {
  // assumes that val is positive
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (val));
  asm("fmant %0, %1": "=r" (mant): "r" (val));
  // mant to SIG_EXP
  right_shift_t shr = SIG_EXP - exp + 23;
  mant >>= shr;
  return mant;
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
  ns.threshold = from_float_pos(th);
  ns.threshold_inv =  0x7fffffffffffffff / ns.threshold ;
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
/*
    invt = utils.int64(((1 << 63) - 1) // threshold_int)

    if -slope_f32 > float32(0) and threshold_int > envelope_int:
        # this looks a bit scary, but as long as envelope < threshold,
        # it can't overflow
        new_gain_int = utils.int64(envelope_int * invt)
        new_gain_int = new_gain_int + 2**31
        new_gain_int = utils.int32(new_gain_int >> 32)
        new_gain_int = ((float32(new_gain_int * 2**-31) ** -slope_f32) * float32(2**31)).as_int32()
    else:
       2**-31 new_gain_int = utils.int32(0x7FFFFFFF)


    if slope_f32 > float32(0) and threshold_int < envelope_int:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
        new_gain_int = ((float32(new_gain_int * 2**-31) ** slope_f32) * float32(2**31)).as_int32()
    else:
        new_gain_int = utils.int32(0x7FFFFFFF)
*/
  int64_t new_gain = 0;
  if (-ns->slope > 0 && ns->threshold > ns->env_det.envelope) {
    // this looks a bit scary, but as long as envelope < threshold,
    // it can't overflow
    new_gain =  ns->env_det.envelope * ns->threshold_inv;
    new_gain += 0x80000000;
    new_gain >>= 32;
    new_gain = (powf(new_gain >> 31, ns->slope)) * (2LL<<31);
  } else {
    new_gain =  0x7fffffffffffffff;
  }
  int32_t alpha = ns->env_det.release_alpha;
  if( ns->gain > new_gain ) {
    alpha = ns->env_det.attack_alpha;
  }

  ns->gain = q31_ema(ns->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, ns->gain);
}
