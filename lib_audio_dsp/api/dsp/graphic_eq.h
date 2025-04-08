// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"


#define Q_GEQ 31

q2_30* adsp_graphic_eq_10b_init(float fs);


/**
 * @brief 10-band graphic equaliser
 * 
 * This function implements an 10-band graphic equalizer filter.
 * The equaliser is implemented as a set of parallel 4th order bandpass
 * filters, with a gain controlling the level of each parallel branch.
 * 
 * @param new_sample    New sample to be filtered
 * @param gains         The gains of each band in Q_GEQ format
 * @param coeffs        Filter coefficients
 * @param state         Filter state, must be DWORD_ALIGNED
 * @return int32_t      Filtered sample
 * @note The filter coefficients can be generated using ``adsp_graphic_eq_10b_init``.
 */
int32_t adsp_graphic_eq_10b(
  int32_t new_sample,
  int32_t gains[10],
  q2_30 coeffs[50],
  int32_t state[160]);