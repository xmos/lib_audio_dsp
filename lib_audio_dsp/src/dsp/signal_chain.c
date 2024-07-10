// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"

#include <xcore/assert.h>

#if Q_GAIN != 27
#error "Need to change the assert value in adsp_dB_to_gain"
#endif

int32_t adsp_dB_to_gain(float dB_gain) {
  xassert(dB_gain <= 24 && "Maximum fixed gain is +24 dB");
  float gain_fl = powf(10, (dB_gain / 20));
  // gain_fl will always be positive
  int32_t zero, exp; unsigned mant;
  asm("fsexp %0, %1, %2": "=r" (zero), "=r" (exp): "r" (gain_fl));
  asm("fmant %0, %1": "=r" (mant): "r" (gain_fl));
  // mant to q27
  right_shift_t shr = -Q_GAIN - exp + 23;
  mant >>= shr;
  return mant;
}

int32_t adsp_from_q31(int32_t input) {
  right_shift_t shr = 31 + SIG_EXP;
  asm("ashr %0, %1, %2": "=r" (input): "r" (input), "r" (shr));
  return input;
}

int32_t adsp_to_q31(int32_t input) {
  // put input into ah so that exp = sig_exp - 32
  // and then to get -31 saturate and extract with sig_exp - 1
  int32_t ah = input, al = 0, pos = Q_SIG + 1;
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (pos), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32":"=r"(ah):"r"(ah),"r"(al),"r"(pos));
  return ah;
}

int32_t adsp_adder(int32_t * input, unsigned n_ch) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so mul all samples by 2 and then lsats and lextract with 1
  int32_t ah = 0, al = 0, mul = 2, one = 1;
  for (unsigned i = 0; i < n_ch; i ++) {
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (input[i]), "r" (mul), "0" (ah), "1" (al));
  }
  // make sure we didn't overflow 32 bits
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
}

int32_t adsp_subtractor(int32_t x, int32_t y) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so mul all samples by 2 and then lsats and lextract with 1
  int32_t ah = 0, al = 0, mul = 2, one = 1;
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (x), "r" (mul), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (y), "r" (-mul), "0" (ah), "1" (al));
  // make sure we didn't overflow 32 bits
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
}

int32_t adsp_fixed_gain(int32_t input, int32_t gain) {
  int32_t q_format = Q_GAIN;
  // adding 1 << (q_format - 1) for rounding
  int32_t ah = 0, al = 1 << (q_format - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (input), "r" (gain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q_format), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q_format));
  return ah;
}

int32_t adsp_mixer(int32_t * input, unsigned n_ch, int32_t gain) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so mul all samples by 2 and then lsats and lextract with 1
  int32_t ah = 0, al = 0, mul = 2, one = 1;
  for (unsigned i = 0; i < n_ch; i ++) {
    int32_t tmp = adsp_fixed_gain(input[i], gain);
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (tmp), "r" (mul), "0" (ah), "1" (al));
  }
  // make sure we didn't overflow 32 bits
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
}

int32_t adsp_saturate_32b(int64_t acc) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so we'll have to use lsats with 1 twice so we can accomodate the whole int64 range
  int32_t ah, al, mask = 0xffffffff, one = 1;
  al = (int32_t)(acc & mask);
  ah = (int32_t)((int64_t)(acc >> 32) & mask);
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  ah <<= one;
  asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (al), "r" (one), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
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

int32_t adsp_volume_control(
  volume_control_t * vol_ctl,
  int32_t samp
) {
  // do the exponential slew
  vol_ctl->gain += (vol_ctl->target_gain - vol_ctl->gain) >> vol_ctl->slew_shift;
  // apply gain
  return adsp_fixed_gain(samp, vol_ctl->gain);
}

void adsp_volume_control_set_gain(
  volume_control_t * vol_ctl,
  int32_t new_gain
) {
  if(!vol_ctl->mute_state) {
    vol_ctl->target_gain = new_gain;
  }
  else
  {
    vol_ctl->saved_gain = new_gain;
  }
}

void adsp_volume_control_mute(
  volume_control_t * vol_ctl
) {
  if (!vol_ctl->mute_state) {
    vol_ctl->mute_state = 1;
    vol_ctl->saved_gain = vol_ctl->target_gain;
    vol_ctl->target_gain = 0;
  }
}

void adsp_volume_control_unmute(
  volume_control_t * vol_ctl
) {
  if (vol_ctl->mute_state) {
    vol_ctl->mute_state = 0;
    vol_ctl->target_gain = vol_ctl->saved_gain;
  }
}

static inline uint32_t _time_to_samples(float fs, float time, time_units_t units) {
  xassert(time >= 0 && "Time has to be positive");
  switch (units) {
    case MILLISECONDS:
      return (uint32_t)(time * fs / 1000);
    case SECONDS:
      return (uint32_t)(time * fs);
    case SAMPLES:
      return (uint32_t)time;
    default:
      xassert(0 && "Invalid time units");
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
  delay.max_delay = _time_to_samples(fs, max_delay, units);
  delay.delay = _time_to_samples(fs, starting_delay, units);
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
  uint32_t new_delay = _time_to_samples(delay->fs, delay_time, units);
  delay->delay = (new_delay <= delay->max_delay) ? new_delay : delay->max_delay;
}

int32_t adsp_delay(
  delay_t * delay,
  int32_t samp
) {
  int32_t out = delay->buffer[delay->buffer_idx];
  delay->buffer[delay->buffer_idx] = samp;
  // Could do this with a modulo operation,
  // but it would break for when the delay is 0
  // and use the division unit
  if (++delay->buffer_idx >= delay->delay) {
    delay->buffer_idx = 0;
  }
  return out;
}
