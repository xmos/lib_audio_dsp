// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/drc.h"

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
 * @brief Initialise a noise suppressor (expander) object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @param ratio               Noise suppression ratio
 * @return noise_suppressor_expander_t Initialised noise suppressor (expander) object
 */
noise_suppressor_expander_t adsp_noise_suppressor_expander_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);

/**
 * @brief Set the threshold of a noise suppressor (expander)
 *
 * @param nse                  Noise suppressor (Expander) object
 * @param new_th              New threshold in Q_SIG
 */
void adsp_noise_suppressor_expander_set_th(
  noise_suppressor_expander_t * nse,
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
 * @brief Initialise a stereo compressor object
 *
 * @param fs                  Sampling frequency
 * @param threshold_db        Threshold in dB
 * @param attack_t            Attack time in seconds
 * @param release_t           Release time in seconds
 * @param ratio               Compression ratio
 * @return compressor_stereo_t       Initialised stereo compressor object
 */
compressor_stereo_t adsp_compressor_rms_stereo_init(
  float fs,
  float threshold_db,
  float attack_t,
  float release_t,
  float ratio);
