// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <xmath/types.h>

/**
 * @brief Envelope detector state structure
 */
typedef struct{
  /** Attack alpha */
  q1_31 attack_alpha;
  /** Release alpha */
  q1_31 release_alpha;
  /** Current envelope */
  int32_t envelope;
}env_detector_t;

/**
 * @brief Limiter state structure
 */
typedef struct{
  /** Envelope detector */
  env_detector_t env_det;
  /** Linear threshold */
  int32_t threshold;
  /** Linear gain */
  int32_t gain;
}limiter_t;

/**
 * @brief Clipper state structure.
 * Should be initilised with the linear threshold
 */
typedef int32_t clipper_t;

/**
 * @brief Noise gate state structure
 */
typedef limiter_t noise_gate_t;

/**
 * @brief Compressor state structure
 */
typedef struct{
  /** Envelope detector */
  env_detector_t env_det;
  /** Linear threshold */
  int32_t threshold;
  /** Linear gain */
  int32_t gain;
  /** Slope of the compression curve */
  float slope;
}compressor_t;

/**
 * @brief Compressor state structure
 */
typedef struct{
  /** Envelope detector */
  env_detector_t env_det_l;
  env_detector_t env_det_r;
  /** Linear threshold */
  int32_t threshold;
  /** Linear gain */
  int32_t gain;
  /** Slope of the compression curve */
  float slope;
}stereo_compressor_t;

typedef struct{
  /** Envelope detector */
  env_detector_t env_det;
  /** Linear threshold */
  int32_t threshold;
  /** Inverse threshold */
  int64_t inv_threshold;
  /** Linear gain */
  int32_t gain;
  /** Slope of the noise suppression curve */
  float slope;
}noise_suppressor_expander_t;

/**
 * @brief Update the envelope detector peak with a new sample
 *
 * @param env_det             Envelope detector object
 * @param new_sample          New sample
 */
void adsp_env_detector_peak(
  env_detector_t * env_det,
  int32_t new_sample);

/**
 * @brief Update the envelope detector RMS with a new sample
 *
 * @param env_det             Envelope detector object
 * @param new_sample          New sample
 */
void adsp_env_detector_rms(
  env_detector_t * env_det,
  int32_t new_sample);

/**
 * @brief Process a new sample with a clipper
 *
 * @param clip                Clipper object
 * @param new_samp            New sample
 * @return int32_t            Clipped sample
 */
int32_t adsp_clipper(
  clipper_t clip,
  int32_t new_samp);

/**
 * @brief Process a new sample with a peak limiter
 *
 * @param lim                 Limiter object
 * @param new_samp            New sample
 * @return int32_t            Limited sample
 */
int32_t adsp_limiter_peak(
  limiter_t * lim,
  int32_t new_samp);

/**
 * @brief Process a new sample with a hard limiter peak
 *
 * @param lim                 Limiter object
 * @param new_samp            New sample
 * @return int32_t            Limited sample
 */
int32_t adsp_hard_limiter_peak(
  limiter_t * lim,
  int32_t new_samp);

/**
 * @brief Process a new sample with an RMS limiter
 *
 * @param lim                 Limiter object
 * @param new_samp            New sample
 * @return int32_t            Limited sample
 */
int32_t adsp_limiter_rms(
  limiter_t * lim,
  int32_t new_samp);

/**
 * @brief Process a new sample with a noise gate
 *
 * @param ng                  Noise gate object
 * @param new_samp            New sample
 * @return int32_t            Gated sample
 */
int32_t adsp_noise_gate(
  noise_gate_t * ng,
  int32_t new_samp);

/**
 * @brief Process a new sample with a noise suppressor (expander)
 *
 * @param nse                 Noise suppressor (Expander) object
 * @param new_samp            New sample
 * @return int32_t            Suppressed sample
 */
int32_t adsp_noise_suppressor_expander(
  noise_suppressor_expander_t * nse,
  int32_t new_samp);

/**
 * @brief Process a new sample with an RMS compressor
 *
 * @param comp                Compressor object
 * @param new_samp            New sample
 * @return int32_t            Compressed sample
 */
int32_t adsp_compressor_rms(
  compressor_t * comp,
  int32_t new_samp);

/**
 * @brief Process a new sample with a sidechain RMS compressor
 *
 * @param comp                Compressor object
 * @param input_samp          Input sample
 * @param detect_samp         Sidechain sample
 * @return int32_t            Compressed sample
 */
int32_t adsp_compressor_rms_sidechain(
  compressor_t * comp,
  int32_t input_samp,
  int32_t detect_samp);

/**
 * @brief Process a pair of new samples with a stereo sidechain RMS compressor
 *
 * @param comp                Compressor object
 * @param outputs_lr          Pointer to the outputs 0:left, 1:right
 * @param input_samp_l        Left input sample
 * @param input_samp_r        Right input sample
 * @param detect_samp_l       Left sidechain sample
 * @param detect_samp_r       Right sidechain sample
 */
void adsp_compressor_rms_sidechain_stereo(
  stereo_compressor_t * comp,
  int32_t outputs_lr[2],
  int32_t input_samp_l,
  int32_t input_samp_r,
  int32_t detect_samp_l,
  int32_t detect_samp_r);
