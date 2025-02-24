// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "control/adsp_control.h"
#include <xcore/assert.h>

#include <math.h>

#if Q_GAIN != 27
#error "Need to change the cap value in adsp_dB_to_gain"
#endif

int32_t adsp_dB_to_gain(float dB_gain) {
  dB_gain = MIN(dB_gain, 24.0f);
  return db_to_qxx(dB_gain, Q_GAIN);
}

volume_control_t adsp_volume_control_init(
  float gain_dB,
  int32_t slew_shift,
  uint8_t mute_state
) {
  volume_control_t vol_ctl;
  vol_ctl.mute_state = mute_state;
  adsp_volume_control_set_gain(&vol_ctl, adsp_dB_to_gain(gain_dB));
  vol_ctl.slew_shift = slew_shift;
  vol_ctl.saved_gain = 0;

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


switch_slew_t adsp_switch_slew_init(float fs, int32_t init_position){
  switch_slew_t out = {.switching = false,
                       .position = init_position,
                       .last_position=init_position,
                       .counter = -(1<<30),
                       .step = INT32_MAX / (int32_t)(fs * 0.03f)};
  return out;
}


void adsp_switch_slew_move(switch_slew_t* switch_slew, int32_t new_position){
  if (new_position != switch_slew->position){
    switch_slew->last_position = switch_slew->position;
    switch_slew->position = new_position;
    switch_slew->switching = true;
    switch_slew->counter = -(1 << 30);
  }
}