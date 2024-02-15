// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"

int32_t adsp_cascaded_biquads_8b(
  int32_t new_sample,
  q2_30 coeffs[40],
  int32_t state[64],
  left_shift_t lsh[8]);

void adsp_design_butterworth_lowpass_8b(
  q2_30 coeffs[40],
  const unsigned N,
  const float fc,
  const float fs);

void adsp_design_butterworth_highpass_8b(
  q2_30 coeffs[40],
  const unsigned N,
  const float fc,
  const float fs);
