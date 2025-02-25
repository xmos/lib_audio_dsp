// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <xcore/assert.h>
#include "control/adsp_control.h"
#include <math.h>

void adsp_reverb_wet_dry_mix(int32_t gains[2], float mix) {
  adsp_crossfader_mix(gains, mix);
  // mix = mix > 1.0f ? 1.0f : mix;
  // mix = mix < 0.0f ? 0.0f : mix;
  // const float pi_by_2 = 1.5707963f;
  // // get an angle [0, pi / 2]
  // float omega = mix * pi_by_2;

  // // -4.5 dB panning
  // float dry = sqrtf((1.0f - mix) * cosf(omega));
  // float wet = sqrtf(mix * sinf(omega));
  // gains[0] = adsp_reverb_float2int(dry);
  // gains[1] = adsp_reverb_float2int(wet);
}

reverb_room_t adsp_reverb_room_init(
  float fs,
  float max_room_size,
  float room_size,
  float decay,
  float damping,
  float wet_gain,
  float dry_gain,
  float pregain,
  float max_predelay,
  float predelay,
  void *reverb_heap)
{
  // For larger rooms, increase max_room_size. Don't forget to also increase
  // the size of reverb_heap
  xassert(room_size >= 0 && room_size <= 1);
  xassert(max_predelay > 0);
  xassert(predelay <= max_predelay);
  xassert(reverb_heap != NULL);

  reverb_room_t rv;

  // Avoids too much or too little feedback
  const int32_t feedback_int = adsp_reverb_calculate_feedback(decay);
  const int32_t damping_int = adsp_reverb_calculate_damping(damping);

  int32_t predelay_samps = predelay * fs / 1000;
  int32_t max_predelay_samps = max_predelay * fs / 1000;
  adsp_reverb_room_init_filters(&rv, fs, max_room_size, max_predelay_samps, predelay_samps, feedback_int, damping_int, reverb_heap);
  adsp_reverb_room_set_room_size(&rv, room_size);

  rv.pre_gain = adsp_reverb_float2int(pregain);
  rv.dry_gain = adsp_reverb_room_calc_gain(dry_gain);
  rv.wet_gain = adsp_reverb_room_calc_gain(wet_gain);

  return rv;
}

static inline void _get_wet_gains(int32_t wet_gains[2], float wet_gain, float width){
  width = width > 1.0f ? 1.0f : width;
  width = width < 0.0f ? 0.0f : width;
  float wet1, wet2;
  wet1 = wet_gain * (width / 2.0f + 0.5f);
  wet2 = wet_gain * ((1.0f - width) / 2.0f);
  wet_gains[0] = adsp_reverb_float2int(wet1);
  wet_gains[1] = adsp_reverb_float2int(wet2);
}

void adsp_reverb_room_st_calc_wet_gains(int32_t wet_gains[2], float wet_gain, float width) {
  wet_gain = powf(10.0f, wet_gain / 20.0f);
  _get_wet_gains(wet_gains, wet_gain, width);
}

void adsp_reverb_st_wet_dry_mix(int32_t gains[3], float mix, float width) {
  mix = mix > 1.0f ? 1.0f : mix;
  mix = mix < 0.0f ? 0.0f : mix;
  const float pi_by_2 = 1.5707963f;
  // get an angle [0, pi / 2]
  float omega = mix * pi_by_2;

  // -4.5 dB panning
  float dry = sqrtf((1.0f - mix) * cosf(omega));
  float wet = sqrtf(mix * sinf(omega));
  gains[0] = adsp_reverb_float2int(dry);
  _get_wet_gains(&gains[1], wet, width);
}

reverb_room_st_t adsp_reverb_room_st_init(
  float fs,
  float max_room_size,
  float room_size,
  float decay,
  float damping,
  float width,
  float wet_gain,
  float dry_gain,
  float pregain,
  float max_predelay,
  float predelay,
  void *reverb_heap)
{
  // For larger rooms, increase max_room_size. Don't forget to also increase
  // the size of reverb_heap
  xassert(room_size >= 0 && room_size <= 1);
  xassert(max_predelay > 0);
  xassert(predelay <= max_predelay);
  xassert(reverb_heap != NULL);

  reverb_room_st_t rv;

  // Avoids too much or too little feedback
  const int32_t feedback_int = adsp_reverb_calculate_feedback(decay);
  const int32_t damping_int = adsp_reverb_calculate_damping(damping);

  int32_t predelay_samps = predelay * fs / 1000;
  int32_t max_predelay_samps = max_predelay * fs / 1000;
  adsp_reverb_room_st_init_filters(&rv, fs, max_room_size, max_predelay_samps, predelay_samps, feedback_int, damping_int, reverb_heap);
  adsp_reverb_room_st_set_room_size(&rv, room_size);

  rv.pre_gain = adsp_reverb_float2int(pregain);
  rv.dry_gain = adsp_reverb_room_calc_gain(dry_gain);
  int32_t wet_gains[2];
  adsp_reverb_room_st_calc_wet_gains(wet_gains, wet_gain, width);
  rv.wet_gain1 = wet_gains[0];
  rv.wet_gain2 = wet_gains[1];

  return rv;
}
