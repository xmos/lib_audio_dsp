// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <xcore/assert.h>
#include "control/adsp_control.h"

#include <math.h>

static inline int32_t float_to_Q_RVR_pos(float val)
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
  right_shift_t shr = -Q_RVR - exp + 23;
  mant >>= shr;
  return mant;
}

int32_t adsp_reverb_room_calc_gain(float gain_db)
{
  xassert(gain_db > ADSP_RVR_MIN_GAIN_DB &&
            gain_db <= ADSP_RVR_MAX_GAIN_DB);
  int32_t gain = float_to_Q_RVR_pos(powf(10, gain_db / 20));
  return gain;
}

void adsp_reverb_wet_dry_mix(int32_t gains[2], float mix) {
  xassert(mix >= 0 && mix <= 1);
  const float pi_by_2 = 1.5707963f;
  // get an angle [0, pi / 2]
  float omega = mix * pi_by_2;

  // -4.5 dB panning
  float dry = sqrtf((1.0f - mix) * cosf(omega));
  float wet = sqrtf(mix * sinf(omega));
  gains[0] = float_to_Q_RVR_pos(dry);
  gains[1] = float_to_Q_RVR_pos(wet);
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
  xassert(decay >= 0 && decay <= 1);
  xassert(damping >= 0 && damping <= 1);
  xassert(pregain >= 0 && pregain < 1);
  xassert(max_predelay > 0);
  xassert(predelay <= max_predelay);
  xassert(reverb_heap != NULL);

  reverb_room_t rv;

  // Avoids too much or too little feedback
  const int32_t feedback_int = float_to_Q_RVR_pos((decay * 0.28) + 0.7);
  const int32_t damping_int = MAX(float_to_Q_RVR_pos(damping) - 1, 1);

  int32_t predelay_samps = predelay * fs / 1000;
  int32_t max_predelay_samps = max_predelay * fs / 1000;
  adsp_reverb_room_init_filters(&rv, fs, max_room_size, max_predelay_samps, predelay_samps, feedback_int, damping_int, reverb_heap);
  adsp_reverb_room_set_room_size(&rv, room_size);

  rv.pre_gain = float_to_Q_RVR_pos(pregain);
  rv.dry_gain = adsp_reverb_room_calc_gain(dry_gain);
  rv.wet_gain = adsp_reverb_room_calc_gain(wet_gain);

  return rv;
}
