// Copyright 2024 XMOS ngITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "dsp/_helpers/drc_utils.h"
#include "print.h"
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
#define POW2_TO_MINUS31 4.656612873077393e-10
noise_suppressor_t adsp_noise_suppressor_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio
) {
  noise_suppressor_t ns;
  printf("threshold_db %f, attack_t %f, release_t %f, ratio %f", threshold_db, attack_t, release_t, ratio);
  ns.env_det = adsp_env_detector_init(fs, attack_t, release_t, 0);
  float th = powf(10, threshold_db / 20);
  ns.threshold = from_float_pos(th);
  ns.threshold_inv =  (int64_t)((float) 0x7fffffffffffffff) / ns.threshold ;
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
        new_gain_int = utils.int32(0x7FFFFFFF)


    if slope_f32 > float32(0) and threshold_int < envelope_int:
        new_gain_int = int(threshold_int) << 31
        new_gain_int = utils.int32(new_gain_int // envelope_int)
        new_gain_int = ((float32(new_gain_int * 2**-31) ** slope_f32) * float32(2**31)).as_int32()
    else:
        new_gain_int = utils.int32(0x7FFFFFFF)
*/
  int32_t new_gain = INT32_MAX;

  if (-ns->slope > 0 && ns->threshold > ns->env_det.envelope) {
    // this looks a bit scary, but as long as envelope < threshold,
    // it can't overflow
    int64_t new_gain_i64 =  ns->env_det.envelope * ns->threshold_inv;
    new_gain_i64 += 0x80000000;
    new_gain = new_gain_i64 >> 32;
    float ns_fl = 0;
    ns_fl = ((float) new_gain) * POW2_TO_MINUS31;
    ns_fl = (powf(ns_fl, -ns->slope));
    //float ns_fl = 0;
    //int32_t r = -Q_alpha + 23;
    //asm("fmake %0, %1, %2, %3, %4": "=r" (ns_fl): "r" (0), "r" (r), "r" (0), "r" (new_gain));
    //ns_fl = (powf(ns_fl, -ns->slope));

    //asm("fsexp %0, %1, %2": "=r" (al), "=r" (r): "r" (ns_fl));
    //asm("fmant %0, %1": "=r" (new_gain): "r" (ns_fl));
    //r = -Q_alpha - r + 23;
    //new_gain >>= r;
    new_gain = (int32_t) (ns_fl * (float) INT32_MAX);
  }
  int32_t alpha = ns->env_det.attack_alpha;
  if( ns->gain > new_gain ) {
    alpha = ns->env_det.release_alpha;
  }

  ns->gain = q31_ema(ns->gain, new_gain, alpha);
  return apply_gain_q31(new_samp, ns->gain);
}
