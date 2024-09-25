// Copyright 2024 XMOS LIMITED.
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
