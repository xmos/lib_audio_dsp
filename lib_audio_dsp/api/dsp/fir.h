// Copyright 2024 XMOS LIMITED.
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

/**
 * @brief Time domain filter struct for reference.
 */
typedef struct td_reference_fir_filter_t{
  /** Pointer to the actual coefficients. */
  int32_t * coefs;
  /** The count of coefficients. */
  uint32_t length;
  /** The output exponent(for printing). */
  uint32_t exponent;
  /** The amount to shr the accumulator after all accumulation is complete. */
  uint32_t accu_shr;
  /** The amount to shr the product of data and coef before accumulating. */
  uint32_t prod_shr;
} td_reference_fir_filter_t;

/**
 * @brief This implements a FIR at the highest possile precision in a human readable way. Its use
 * is for debug and regression.
 * 
 * @param new_sample A single sample to add to the time series data.
 * @param filter Pointer to the td_reference_fir_filter_t struct.
 * @param data Pointer to the actual time series data.
 * @return int32_t The output of the filtered data.
 */
int32_t td_reference_fir(
    int32_t new_sample,
    td_reference_fir_filter_t * filter,
    int32_t * data);