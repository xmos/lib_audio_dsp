// Copyright 2024 XMOS LIMITED.
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
 * @brief Clipper state structure
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
}expander_t;

/**
 * @brief Initialise an envelope detector object
 *
 * @param fs                  Sampling frequency
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @return env_detector_t     Initialised envelope detector object
 * @note Detect time is optional. If specified, attack and release times will be equal.
 */
env_detector_t adsp_env_detector_init(
  float fs,
  float attack_t,
  float release_t);

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
 * @brief Initialise a (hard) limiter peak object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @return limiter_t          Initialised limiter object
 */
limiter_t adsp_limiter_peak_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t);

/**
 * @brief Initialise an RMS limiter object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @return limiter_t          Initialised limiter object
 */
limiter_t adsp_limiter_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t);

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
 * @brief Initialise a noise gate object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @return noise_gate_t       Initialised noise gate object
 */
noise_gate_t adsp_noise_gate_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t);

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
 * @brief Initialise an expander object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @param ratio               Noise suppression ratio
 * @return expander_t Initialised expander object
 */
expander_t adsp_expander_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);

/**
 * @brief Process a new sample with an expander
 *
 * @param ex                  Expander object
 * @param new_samp            New sample
 * @return int32_t            Suppressed sample
 */
int32_t adsp_expander(
  expander_t * ex,
  int32_t new_samp);

/**
 * @brief Set the threshold of an expander
 *
 * @param ex                  Expander object
 * @param new_th              New threshold
 */
void adsp_expander_set_th(
  expander_t * ex,
  int32_t new_th);

/**
 * @brief Initialise a compressor object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @param ratio               Compression ratio
 * @return compressor_t       Initialised compressor object
 */
compressor_t adsp_compressor_rms_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);

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
