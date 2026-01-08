// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/signal_chain.h"

/**
 * @brief Enum for different time units
 */
typedef enum{
  /** Time in samples */
  SAMPLES = 0,
  /** Time in milliseconds */
  MILLISECONDS = 1,
  /** Time in seconds */
  SECONDS = 2
} time_units_t;

/**
 * @brief Convert dB gain to linear gain.
 *
 * @param dB_gain           Gain in dB
 * @return int32_t          Linear gain in Q_GAIN format
 * @note With the current Q_GAIN format, the maximum gain is +24 dB, dB_gain will be saturated to this value
 * @note Passing -INFINITY to this function will give a linear gain of 0.
 */
int32_t adsp_dB_to_gain(float dB_gain);

/**
 * @brief Initialise a slewing gain object
 *
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
 * @param init_gain         Initial gain
 * @param slew_shift        Shift value used in the exponential slew
 * @return gain_slew_t   The slewing gain object.
 */
gain_slew_t adsp_slew_gain_init(int32_t init_gain, int32_t slew_shift);

/**
 * @brief Initialise volume control object.
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
 * @param mute_state        Initial mute state
 * @return volume_control_t Volume control state object
 */
volume_control_t adsp_volume_control_init(
  float gain_dB,
  int32_t slew_shift,
  uint8_t mute_state);

/**
 * @brief Initialise a delay object
 *
 * @param fs                Sampling frequency
 * @param max_delay         Maximum delay in specified units
 * @param starting_delay    Initial delay in specified units
 * @param units             Time units (SAMPLES, MILLISECONDS, SECONDS). If an invalid
 *                          unit is passed, SAMPLES is used.
 * @param delay_heap        Pointer to the allocated delay memory
 * @return delay_t          Delay state object
 */
delay_t adsp_delay_init(
  float fs,
  float max_delay,
  float starting_delay,
  time_units_t units,
  void * delay_heap);

/**
 * @brief Set the delay of a delay object.
 * Will set the delay to the new value, saturating to the maximum delay
 *
 * @param delay             Delay object
 * @param delay_time        New delay time in specified units
 * @param units             Time units (SAMPLES, MILLISECONDS, SECONDS). If an invalid
 *                          unit is passed, SAMPLES is used.
 */
void adsp_set_delay(
  delay_t * delay,
  float delay_time,
  time_units_t units);

/**
 * @brief Convert a time in seconds/milliseconds/samples to samples for a
 * given sampling frequency.
 *
 * @param fs                Sampling frequency
 * @param time              New delay time in specified units
 * @param units             Time units (SAMPLES, MILLISECONDS, SECONDS) . If an invalid
 *                          unit is passed, SAMPLES is used.
 * @return uint32_t         Time in samples
 */
uint32_t time_to_samples(float fs, float time, time_units_t units);

/**
 * @brief Initialise a slewing switch object
 *
 * @param fs               Sampling frequency, used to calculate the
 *                         step size.
 * @param init_position    Starting position of the switch.
 * @return switch_slew_t   The slewing switch object.
 */
switch_slew_t adsp_switch_slew_init(float fs, int32_t init_position);

/**
 * @brief Move the position of the switch. This sets the state of the
 * switch for slewing on subsequent samples.
 * 
 * @param switch_slew     Slewing switch state object.
 * @param new_position    The desired input channel to switch to.
 */
void adsp_switch_slew_move(switch_slew_t* switch_slew, int32_t new_position);

/**
 * @brief Calculate the gains for a crossfader according to the mix amount.
 * 
 * When the mix is set to 0, only the first signal will be output. 
 * gains[0] will be max and gains[1] will be 0.
 * When the mix is set to 1, only they second signal will be output. 
 * gains[0] will be 0 and gains[1] will be max.
 * In order to maintain a consistent signal level across all mix values, 
 * when the mix is set to 0.5, each channel has a gain of -4.5 dB.
 *
 * @param gains           Output gains
 * @param mix             Mix applied from 0 to 1
 */
void adsp_crossfader_mix(int32_t gains[2], float mix);
