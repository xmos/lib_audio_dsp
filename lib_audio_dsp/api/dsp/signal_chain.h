// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

/**
 * @brief Volume control state structure
 */
typedef struct{
  // Target linear gain
  int32_t target_gain;
  // Current linear gain
  int32_t gain;
  // Slew shift
  int32_t slew_shift;
  // Saved linear gain
  int32_t saved_gain;
  // Mute flag
  uint8_t mute;
}volume_control_t;

/**
 * @brief Convert dB gain to linear gain
 * 
 * @param dB_gain           Gain in dB
 * @return int32_t          Linear gain in Q_GAIN format
 * @note With the current Q_GAIN format, the maximum gain is +24 dB
 */
int32_t adsp_dB_to_gain(float dB_gain);

/**
 * @brief Convert from Q31 to Q_SIG
 * 
 * @param input             Input in Q31 format
 * @return int32_t          Output in Q_SIG format
 */
int32_t adsp_from_q31(int32_t input);

/**
 * @brief Convert from Q_SIG to Q31
 * 
 * @param input             Input in Q_SIG format
 * @return int32_t          Output in Q31 format
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
 * @brief Saturating subtraction of two samples
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
 * @brief Mixer
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
 * @brief Saturating 64-bit accumulator
 * Will saturate to 32-bit, so that the output value is in the range of int32_t
 * 
 * @param acc               Accumulator
 * @return int32_t          Saturated value
 */
int32_t adsp_saturate_32b(int64_t acc);

/**
 * @brief Initialise volume control object
 * The slew shift will determine the speed of the volume change.
 * A list of the first 10 slew shifts is shown below:
 * 
 * 1  ->  0.03 ms,
 * 2  ->  0.07 ms,
 * 3  ->  0.16 ms,
 * 4  ->  0.32 ms,
 * 5  ->  0.66 ms,
 * 6  ->  1.32 ms,
 * 7  ->  2.66 ms,
 * 8  ->  5.32 ms,
 * 9  -> 10.66 ms,
 * 10 -> 21.32 ms.
 * 
 * @param gain_dB           Target gain in dB
 * @param slew_shift        Shift value used in the exponential slew
 * @return volume_control_t Volume control state object
 */
volume_control_t adsp_volume_control_init(
  float gain_dB,
  int32_t slew_shift);

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
 * @brief Mute a volume control
 * Will save the current target gain and set the taget gain to 0
 * 
 * @param vol_ctl           Volume control object
 */
void adsp_volume_control_mute(
  volume_control_t * vol_ctl);

/**
 * @brief Unmute a volume control
 * Will restore the saved target gain
 * 
 * @param vol_ctl           Volume control object
 */
void adsp_volume_control_unmute(
  volume_control_t * vol_ctl);
