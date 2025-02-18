// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"

/**
 * @brief Slewing biquad state structure
 */
typedef struct {
  /** Target filter coefficients, the active coefficients are slewed towards these */
  q2_30 DWORD_ALIGNED target_coeffs[8];
  /** Active filter coefficients, used to filter the audio */
  q2_30 DWORD_ALIGNED active_coeffs[8];
  /** Left shift compensation for if the filter coefficents are large
   *  and cannot be represented in Q1.30, must be positive */
  left_shift_t lsh;
  /** Shift value used by the exponential slew */
  int32_t slew_shift;
  /** Remaining shifts for cases when the left shift changes during a target_coeff update. */
  left_shift_t remaining_shifts;

} biquad_slew_t;


/**
 * @brief Biquad filter.
 *  This function implements a biquad filter. The filter is implemented as a direct form 1
 * 
 * @param new_sample      New sample to be filtered
 * @param coeffs          Filter coefficients
 * @param state           Filter state
 * @param lsh             Left shift compensation value, must be positive
 * @return int32_t        Filtered sample
 */
int32_t adsp_biquad(
  int32_t new_sample,
  q2_30 coeffs[5],
  int32_t state[8],
  left_shift_t lsh);


/**
 * @brief Slew the active filter coefficients towards the target filter
 * coefficients. This function should be called either once per sample or
 * per frame, and before calling ``adsp_biquad`` to do the filtering.
 * 
 * @param slew_state       Slewing biquad state object
 * @param states           Filter state for each biquad channel
 * @param channels         Number of channels in states
 */
void adsp_biquad_slew_coeffs(
  biquad_slew_t* slew_state,
  int32_t** states,
  int32_t channels
);
