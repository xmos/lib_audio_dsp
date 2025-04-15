// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/filter.h"

/** Heap size to allocate for the delay from samples */
#define FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(SAMPLES) (sizeof(int32_t) * (SAMPLES))

/**
 * @brief Delay state structure
 */
typedef struct{
  // lib_xcore_math FIR struct
  filter_fir_s32_t filter;
} fir_direct_t;
