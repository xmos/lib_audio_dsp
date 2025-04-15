// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <xcore/assert.h>
#include "control/adsp_control.h"
#include <math.h>

reverb_plate_t adsp_reverb_plate_init(
  float fs,
  float decay,
  float damping,
  float bandwidth,
  float early_diffusion,
  float late_diffusion,
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
  int32_t bandwidth_int = adsp_reverb_plate_calc_bandwidth(bandwidth, fs);
  rv.lowpasses[0] = lowpass_1ord_init(bandwidth_int);
  int32_t damping_int = adsp_reverb_plate_calc_damping(damping);
  rv.lowpasses[1] = lowpass_1ord_init(damping_int);
  rv.lowpasses[2] = lowpass_1ord_init(damping_int);

  rv.pre_gain = adsp_reverb_float2int(pregain);
  rv.dry_gain = adsp_reverb_room_calc_gain(dry_gain);
  int32_t wet_gains[2];
  adsp_reverb_room_st_calc_wet_gains(wet_gains, wet_gain, width);
  rv.wet_gain1 = wet_gains[0];
  rv.wet_gain2 = wet_gains[1];

  rv.decay = adsp_reverb_float2int(decay);
  decay += 0.15;
  decay = (decay < 0.25) ? 0.25 : decay;
  decay = (decay > 0.5) ? 0.5 : decay;
  int32_t decay_diff_2 = adsp_reverb_float2int(decay);
  int32_t decay_diff_1 = adsp_reverb_plate_calc_late_diffusion(late_diffusion);
  int32_t in_dif1 = adsp_reverb_float2int(early_diffusion);
  int32_t in_dif2 = adsp_reverb_float2int(early_diffusion * 5 / 6);
  int32_t predelay_samps = predelay * fs / 1000;
  int32_t max_predelay_samps = max_predelay * fs / 1000;

  adsp_reverb_plate_init_filters(&rv, fs, decay_diff_1, decay_diff_2, in_dif1, in_dif2, max_predelay_samps, predelay_samps, reverb_heap);
  return rv;
}
