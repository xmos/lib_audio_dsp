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


typedef struct {
  q2_30 DWORD_ALIGNED target_coeffs[8];
  q2_30 DWORD_ALIGNED coeffs[8];
  int32_t slew_shift;
  left_shift_t remaining_shifts;
  left_shift_t lsh;
} biquad_slew_t;


void adsp_biquad_slew_init(
  biquad_slew_t* slew_state,
  q2_30 target_coeffs[8],
  left_shift_t lsh,
  left_shift_t slew_shift
);

void adsp_biquad_slew_update(
  biquad_slew_t* slew_state,
  int32_t** states,
  int32_t channels,
  q2_30 target_coeffs[8],
  left_shift_t lsh
);

void adsp_biquad_slew_coeffs(
  biquad_slew_t* slew_state,
  int32_t** states,
  int32_t channels
);
