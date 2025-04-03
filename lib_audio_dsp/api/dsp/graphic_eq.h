// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"


int32_t* adsp_graphic_eq_10b_init(float fs);



/**
 * @brief 8-band cascaded biquad filter
 * This function implements an 8-band cascaded biquad filter. The filter is implemented as a direct
 * form 1 filter.
 * 
 * @param new_sample    New sample to be filtered
 * @param coeffs        Filter coefficients
 * @param state         Filter state
 * @return int32_t      Filtered sample
 * @note The filter coefficients must be in [5][8]
 */
int32_t adsp_graphic_eq_10b(
  int32_t new_sample,
  int32_t gains[10],
  q2_30 coeffs[50],
  int32_t state[80]);