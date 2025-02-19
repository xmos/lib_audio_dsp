// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"

#include <xcore/assert.h>

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

int32_t _cos_approx(int32_t x){
  // approximate a cosine fade out with a 2 term polynomial
  int32_t x2 = ((int64_t)x*x) >> 30;
  
  int32_t y = -1622688857;
  y += ((int64_t)x2*549248075) >> 30;
  y = ((int64_t)x*y) >> 30;
  y += 1 << 30;

  return y;
}


switch_slew_t adsp_switch_slew_init(int32_t fs, int32_t init_position){
  switch_slew_t out = {.switching = false,
                       .position = init_position,
                       .last_position=init_position,
                       .counter = -(1<<30),
                       .step = INT32_MAX / (int32_t)(fs * 0.03f)};
  return out;
}


int32_t adsp_switch_slew(switch_slew_t* switch_slew, int32_t* samples){

  if (switch_slew->switching){
    int32_t gain_1 = _cos_approx(switch_slew->counter);
    int32_t y = ((int64_t)gain_1 * samples[switch_slew->last_position]) >> 31;
    int32_t gain_2 = INT32_MAX - gain_1;
    y += ((int64_t)gain_2 * samples[switch_slew->position]) >> 31;

    switch_slew->counter += switch_slew->step;
    if (switch_slew->counter > 1 <<30){
      switch_slew->switching = false;
    }

    return y;
  }
  else{
    return samples[switch_slew->position];
    }
  }

void adsp_switch_slew_move(switch_slew_t* switch_slew, int32_t new_position){
  if (new_position != switch_slew->position){
    switch_slew->last_position = switch_slew->position;
    switch_slew->position = new_position;
    switch_slew->switching = true;
    switch_slew->counter = -(1 << 30);
  }
}