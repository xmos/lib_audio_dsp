// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"

/**
 * @brief Biquad filter.
 *  This function implements a biquad filter. The filter is implemented as a direct form 1
 * 
 * @param new_sample      New sample to be filtered
 * @param coeffs          Filter coefficients
 * @param state           Filter state
 * @param lsh             Left shift compensation value
 * @return int32_t        Filtered sample
 */
int32_t adsp_biquad(
  int32_t new_sample,
  q2_30 coeffs[5],
  int32_t state[8],
  left_shift_t lsh);


/**
 * @brief Biquad filter with slew.
 *  This function implements a biquad filter with slew. 
 *  The filter is implemented as a direct form 1.
 *  The coeffs are exponentially slewed towards the target_coeffs.
 *  This is can be used for real time adjustable biquads.
 * 
 * @param new_sample      New sample to be filtered
 * @param coeffs          Filter coefficients
 * @param target_coeffs   Target filter coefficients
 * @param state           Filter state
 * @param lsh             Left shift compensation value
 * @param slew_shift      Shift value used in the exponential slew
 * @return int32_t        Filtered sample
 */
int32_t adsp_biquad_slew(
  int32_t new_sample,
  q2_30 coeffs[8],
  q2_30 target_coeffs[8],
  int32_t state[8],
  left_shift_t lsh,
  int32_t slew_shift);