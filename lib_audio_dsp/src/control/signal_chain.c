// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "control/adsp_control.h"
#include <xcore/assert.h>

#include <math.h>

#if Q_GAIN != 27
#error "Need to change the assert value in adsp_dB_to_gain"
#endif

#ifndef __XS3A__
/**
 * @brief Convert a positive float value to a fixed point int32 number in
 *        Q_SIG format. By assuming the value is positive (e.g. a gain value
 *        converted from decibels), negative cases can be ignored. If the
 *        value of x exceeds the fixed point maximum, it is saturated.
 * 
 * @param x A positive floating point value
 * @return int32_t x in Q_SIG fixed point format
 */
static inline int32_t _positive_float2fixed_qgain(float x)
{
  if ( x >= 1 << (31-Q_GAIN)) return INT32_MAX;
  else if( x > 0 ) return (((float)((1 << Q_GAIN) - 1)) * x + 0.5f);
  return 0;
}
#endif


int32_t adsp_dB_to_gain(float dB_gain) {
  // special case for -inf dB
  if (dB_gain == -INFINITY){return 0;}

  dB_gain = MIN(dB_gain, 24.0f);
  float gain_fl = powf(10.0f, (dB_gain / 20.0f));
  
  #ifdef __XS3A__
    // gain_fl will always be positive
    int32_t zero, exp; unsigned mant;
    asm("fsexp %0, %1, %2": "=r" (zero), "=r" (exp): "r" (gain_fl));
    asm("fmant %0, %1": "=r" (mant): "r" (gain_fl));
    // mant to q27
    right_shift_t shr = -Q_GAIN - exp + 23;
    mant >>= shr;
  #else
    int32_t mant = _positive_float2fixed_qgain(A);
  #endif
  return mant;
}

volume_control_t adsp_volume_control_init(
  float gain_dB,
  int32_t slew_shift,
  uint8_t mute_state
) {
  volume_control_t vol_ctl;
  
  vol_ctl.mute_state = mute_state;

  // if starting muted, we need to set target_gain to 0.
  if (mute_state){
    adsp_volume_control_mute(&vol_ctl);
  }

  adsp_volume_control_set_gain(&vol_ctl, adsp_dB_to_gain(gain_dB));

  vol_ctl.slew_shift = slew_shift;

  return vol_ctl;
}

uint32_t time_to_samples(float fs, float time, time_units_t units) {
  time = MAX(time, 0); // Time has to be positive
  switch (units) {
    case MILLISECONDS:
      return (uint32_t)(time * fs / 1000);
    case SECONDS:
      return (uint32_t)(time * fs);
    case SAMPLES:
      return (uint32_t)time;
    default:
      xassert(0);  // Invalid time units

  }
}

delay_t adsp_delay_init(
  float fs,
  float max_delay,
  float starting_delay,
  time_units_t units,
  void * delay_heap
) {
  delay_t delay;
  delay.fs = fs;
  xassert(delay.max_delay > 0 && "Max delay must be greater than 0");
  delay.max_delay = time_to_samples(fs, max_delay, units);
  delay.delay = time_to_samples(fs, starting_delay, units);
  xassert(delay.delay <= delay.max_delay && "Starting delay must be less or equal to the max delay");
  xassert(delay_heap != NULL && "Delay heap must be allocated");
  delay.buffer_idx = 0;
  delay.buffer = (int32_t *)delay_heap;
  return delay;
}

void adsp_set_delay(
  delay_t * delay,
  float delay_time,
  time_units_t units
) {
  uint32_t new_delay = time_to_samples(delay->fs, delay_time, units);
  delay->delay = (new_delay <= delay->max_delay) ? new_delay : delay->max_delay;
}
