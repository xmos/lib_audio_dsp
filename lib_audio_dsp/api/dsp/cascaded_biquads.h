// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"

/**
 * @brief 8-band cascaded biquad filter
 * This function implements an 8-band cascaded biquad filter. The filter is implemented as a direct
 * form 1 filter.
 * 
 * @param new_sample    New sample to be filtered
 * @param coeffs        Filter coefficients
 * @param state         Filter state
 * @param lsh           Left shift compensation value
 * @return int32_t      Filtered sample
 * @note The filter coefficients must be in [5][8]
 */
int32_t adsp_cascaded_biquads_8b(
  int32_t new_sample,
  q2_30 coeffs[40],
  int32_t state[64],
  left_shift_t lsh[8]);

/**
 * @brief 16-band cascaded biquad filter
 * This function implements a 16-band cascaded biquad filter. The filter is implemented as a direct
 * form 1 filter.
 * 
 * @param new_sample    New sample to be filtered
 * @param coeffs        Filter coefficients
 * @param state         Filter state
 * @param lsh           Left shift compensation value
 * @return int32_t      Filtered sample
 * @note The filter coefficients must be in [5][16]
 */
int32_t adsp_cascaded_biquads_16b(
  int32_t new_sample,
  q2_30 coeffs[80],
  int32_t state[128],
  left_shift_t lsh[16]);
