// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

typedef struct{
  q1_31 attack_alpha;
  q1_31 release_alpha;
  int32_t envelope;
}env_detector_t;

typedef struct{
  env_detector_t env_det;
  int32_t threshold;
  int32_t gain;
}limiter_t;

typedef limiter_t noise_gate_t;

typedef struct{
  env_detector_t env_det;
  int32_t threshold;
  int32_t gain;
  float slope;
}compressor_t;

typedef struct{
  env_detector_t env_det;
  int32_t threshold;
  int64_t inv_threshold;
  int32_t gain;
  float slope;
}noise_suppressor_t;

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
  float attack_t,
  float release_t);

limiter_t adsp_limiter_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t);

int32_t adsp_limiter_peak(
  limiter_t * lim,
  int32_t new_samp);

int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp);

noise_gate_t adsp_noise_gate_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t);

int32_t adsp_noise_gate(
  noise_gate_t * ng,
  int32_t new_samp);

noise_suppressor_t adsp_noise_suppressor_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);

int32_t adsp_noise_suppressor(
  noise_suppressor_t * ns,
  int32_t new_samp);

void adsp_noise_suppressor_set_th(
  noise_suppressor_t * ns,
  int32_t new_th);

compressor_t adsp_compressor_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);

int32_t adsp_compressor_rms(
  compressor_t * comp,
  int32_t new_samp);

int32_t adsp_compressor_rms_sidechain(
  compressor_t * comp,
  int32_t input_samp,
  int32_t detect_samp);
