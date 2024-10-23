// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <xcore/assert.h>
#include "control/adsp_control.h"

#include <math.h>

static inline int32_t float_to_Q_RVP_pos(float val)
{
  // only works for positive values
  xassert(val >= 0);
  if (val == 1.0f) {
    return INT32_MAX;
  } else if (val == 0.0f) {
    return 0;
  }
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r"(sign), "=r"(exp): "r"(val));
  asm("fmant %0, %1": "=r"(mant): "r"(val));
  // mant to q_rvr
  right_shift_t shr = -Q_RVP - exp + 23;
  mant >>= shr;
  return mant;
}

reverb_plate_t adsp_reverb_plate_init(
  float fs,
  float decay,
  float damping,
  float diffusion,
  float bandwidth,
  float in_diffusion_1,
  float in_diffusion_2,
  float width,
  float wet_gain,
  float dry_gain,
  float pregain,
  float max_predelay,
  float predelay,
  void * reverb_heap)
{
  xassert(reverb_heap != NULL);
  reverb_plate_t rv;
  // lowpasses
  int32_t bandwidth_int = float_to_Q_RVP_pos(bandwidth);
  rv.lowpasses[0] = lowpass_1ord_init(bandwidth_int);
  int32_t damping_int = float_to_Q_RVP_pos(damping);
  rv.lowpasses[1] = lowpass_1ord_init(INT32_MAX - damping_int);
  rv.lowpasses[2] = lowpass_1ord_init(INT32_MAX - damping_int);

  rv.pre_gain = float_to_Q_RVP_pos(pregain);
  rv.dry_gain = adsp_reverb_room_calc_gain(dry_gain);
  int32_t wet_gains[2];
  adsp_reverb_room_st_calc_wet_gains(wet_gains, wet_gain, width);
  rv.wet_gain1 = wet_gains[0];
  rv.wet_gain2 = wet_gains[1];

  decay += 0.15;
  decay = (decay < 0.25) ? 0.25 : decay;
  decay = (decay > 0.5) ? 0.5 : decay;
  int32_t decay_int = float_to_Q_RVP_pos(decay);
  int32_t diffusion_int = -float_to_Q_RVP_pos(diffusion);
  int32_t in_dif1 = float_to_Q_RVP_pos(in_diffusion_1);
  int32_t in_dif2 = float_to_Q_RVP_pos(in_diffusion_2);
  int32_t predelay_samps = predelay * fs / 1000;
  int32_t max_predelay_samps = max_predelay * fs / 1000;

  adsp_reverb_plate_init_filters(&rv, fs, decay_int, diffusion_int, in_dif1, in_dif2, max_predelay_samps, predelay_samps, reverb_heap);
  return rv;
}
