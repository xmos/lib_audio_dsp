// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

typedef struct{
  float attack_alpha;
  float release_alpha;
  float envelope;
}env_detector_t;

typedef struct{
  env_detector_t env_det;
  float threshold;
  float gain;
}limiter_t;

env_detector_t adsp_env_detector_init(
  float fs,
  float attack_t,
  float release_t,
  float detect_t);

void adsp_env_detector_peak(
  env_detector_t * env_det,
  int32_t new_sample);

void adsp_env_detector_rms(
  env_detector_t * env_det,
  int32_t new_sample);

limiter_t adsp_limiter_peak_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t);

limiter_t adsp_limiter_rms_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t);

int32_t adsp_limiter_peak(
  limiter_t * lim,
  int32_t new_samp);

int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp);
