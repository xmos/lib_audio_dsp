// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

/** Heap size to allocate for the delay from samples */
#define DELAY_DSP_REQUIRED_MEMORY_SAMPLES(SAMPLES) (sizeof(int32_t) * (SAMPLES))
/** Heap size to allocate for the delay from milliseconds */
#define DELAY_DSP_REQUIRED_MEMORY_MS(FS, MS) (sizeof(int32_t) * ((FS) * (MS) / 1000))
/** Heap size to allocate for the delay from seconds */
#define DELAY_DSP_REQUIRED_MEMORY_SEC(FS, SEC) (sizeof(int32_t) * (FS) * (SEC))

/** Gain format to be used in the gain APIs */
#define Q_GAIN 27

/**
 * @brief Volume control state structure
 */
typedef struct{
  /** Target linear gain */
  int32_t target_gain;
  /** Current linear gain */
  int32_t gain;
  /** Slew shift */
  int32_t slew_shift;
  /** Saved linear gain */
  int32_t saved_gain;
  /** Mute state: 0: unmuted, 1 muted */
  uint8_t mute_state;
}volume_control_t;

/**
 * @brief Delay state structure
 */
typedef struct{
  /** Sampling frequency */
  float fs;
  /** Current delay in samples */
  uint32_t delay;
  /** Maximum delay in samples */
  uint32_t max_delay;
  /** Current buffer index */
  uint32_t buffer_idx;
  /** Buffer */
  int32_t * buffer;
} delay_t;

/**
 * @brief Convert from Q0.31 to Q_SIG
 *
 * @param input             Input in Q0.31 format
 * @return int32_t          Output in Q_SIG format
 */
int32_t adsp_from_q31(int32_t input);

/**
 * @brief Convert from Q_SIG to Q0.31
 *
 * @param input             Input in Q_SIG format
 * @return int32_t          Output in Q0.31 format
 */
int32_t adsp_to_q31(int32_t input);

/**
 * @brief Saturating addition of an array of samples
 *
 * @param input             Array of samples
 * @param n_ch              Number of channels
 * @return int32_t          Sum of samples
 * @note Will work for any q format
 */
int32_t adsp_adder(int32_t * input, unsigned n_ch);

/**
 * @brief Saturating subtraction of two samples, this returns `x - y`.
 *
 * @param x                 Minuend
 * @param y                 Subtrahend
 * @return int32_t          Difference
 * @note Will work for any q format
 */
int32_t adsp_subtractor(int32_t x, int32_t y);

/**
 * @brief Fixed-point gain
 *
 * @param input             Input sample
 * @param gain              Gain
 * @return int32_t          Output sample
 * @note One of the inputs has to be in Q_GAIN format
 */
int32_t adsp_fixed_gain(int32_t input, int32_t gain);

/**
 * @brief Mixer.
 * Will add signals with gain applied to each signal before mixing
 *
 * @param input             Array of samples
 * @param n_ch              Number of channels
 * @param gain              Gain
 * @return int32_t          Mixed sample
 * @note Inputs or gain have to be in Q_GAIN format
 */
int32_t adsp_mixer(int32_t * input, unsigned n_ch, int32_t gain);

/**
 * @brief Saturating 64-bit accumulator.
 * Will saturate to 32-bit, so that the output value is in the range of int32_t
 *
 * @param acc               Accumulator
 * @return int32_t          Saturated value
 */
int32_t adsp_saturate_32b(int64_t acc);

/**
 * @brief Process a new sample with a volume control
 *
 * @param vol_ctl           Volume control object
 * @param samp              New sample
 * @return int32_t          Processed sample
 */
int32_t adsp_volume_control(
  volume_control_t * vol_ctl,
  int32_t samp);

/**
 * @brief Set the target gain of a volume control
 *
 * @param vol_ctl           Volume control object
 * @param new_gain          New target linear gain
 */
void adsp_volume_control_set_gain(
  volume_control_t * vol_ctl,
  int32_t new_gain);

/**
 * @brief Mute a volume control.
 * Will save the current target gain and set the target gain to 0
 *
 * @param vol_ctl           Volume control object
 */
void adsp_volume_control_mute(
  volume_control_t * vol_ctl);

/**
 * @brief Unmute a volume control.
 * Will restore the saved target gain
 *
 * @param vol_ctl           Volume control object
 */
void adsp_volume_control_unmute(
  volume_control_t * vol_ctl);

/**
 * @brief Process a new sample through a delay object
 * 
 * @note The minimum delay provided by this block is 1 sample. Setting
 *       the delay to 0 will still yield a 1 sample delay.
 *
 * @param delay             Delay object
 * @param samp              New sample
 * @return int32_t          Oldest sample
 */
int32_t adsp_delay(
  delay_t * delay,
  int32_t samp);

