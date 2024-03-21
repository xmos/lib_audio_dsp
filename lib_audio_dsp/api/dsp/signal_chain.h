// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

typedef struct{
  int32_t target_gain;
  int32_t gain;
  int32_t slew_shift;
  int32_t saved_gain;
  uint8_t mute;
}volume_control_t;

int32_t adsp_dB_to_gain(float dB_gain);

int32_t adsp_from_q31(int32_t input);

int32_t adsp_to_q31(int32_t input);

int32_t adsp_adder(int32_t * input, unsigned n_ch);

int32_t adsp_subtractor(int32_t x, int32_t y);

int32_t adsp_fixed_gain(int32_t input, int32_t gain);

int32_t adsp_mixer(int32_t * input, unsigned n_ch, int32_t gain);

int32_t adsp_saturate_32b(int64_t acc);

volume_control_t adsp_volume_control_init(
  float gain_dB,
  int32_t slew_shift);

int32_t adsp_volume_control(
  volume_control_t * vol_ctl,
  int32_t samp);

void adsp_volume_control_set_gain(
  volume_control_t * vol_ctl,
  int32_t new_gain);

void adsp_volume_control_mute(
  volume_control_t * vol_ctl);

void adsp_volume_control_unmute(
  volume_control_t * vol_ctl);
