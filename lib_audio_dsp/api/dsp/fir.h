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


typedef struct td_reference_fir_filter_t{
    int32_t * coefs; //a pointer to the actual coefficients
    uint32_t length; //the count of coefficients
    uint32_t exponent; //the output exponent(for printing)
    uint32_t accu_shr; //the amount to shr the accumulator after all accumulation is complete
    uint32_t prod_shr; //the amount to shr the product of data and coef before accumulating
} td_reference_fir_filter_t;

int32_t td_reference_fir(
    int32_t new_sample,
    td_reference_fir_filter_t * filter,
    int32_t * data);