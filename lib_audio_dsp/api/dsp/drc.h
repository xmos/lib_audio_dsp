// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

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
}noise_suppressor_t;

/**
 * @brief Initialise an envelope detector object
 * 
 * @param fs                  Sampling frequency
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @param detect_t            Detection time in seconds
 * @return env_detector_t     Initialised envelope detector object
 * @note Detect time is optional. If specified, attack and release times will be equal.
 */
env_detector_t adsp_env_detector_init(
  float fs,
  float attack_t,
  float release_t,
  float detect_t);

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
 * @brief Initialise a peak limiter object
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
 * @brief Initialise a noise suppressor object
 * 
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @param ratio               Noise suppression ratio
 * @return noise_suppressor_t Initialised noise suppressor object
 */
noise_suppressor_t adsp_noise_suppressor_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);

/**
 * @brief Process a new sample with a noise suppressor
 * 
 * @param ns                  Noise suppressor object
 * @param new_samp            New sample
 * @return int32_t            Suppressed sample
 */
int32_t adsp_noise_suppressor(
  noise_suppressor_t * ns,
  int32_t new_samp);

/**
 * @brief Set the threshold of a noise suppressor
 * 
 * @param ns                  Noise suppressor object
 * @param new_th              New threshold
 */
void adsp_noise_suppressor_set_th(
  noise_suppressor_t * ns,
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
