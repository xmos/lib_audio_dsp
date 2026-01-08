// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/filter.h"
#include "dsp/td_block_fir.h"
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

/**
 * @brief Function to add samples to the FIR data structure. This is for debug and test only.
 * 
 * @param input_block Array of int32_t samples of length TD_BLOCK_FIR_LENGTH.
 * @param fir_data Pointer to struct of type td_block_fir_data_t to which the samples will be added.
 */
void td_block_fir_add_data_ref(
    int32_t input_block[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data);
    
/**
 * @brief Function to compute the convolution between fir_data and fir_filter. This is for debug and test only.
 * 
 * @param samples_out Array of length TD_BLOCK_FIR_LENGTH(8), which will be used to return the 
        processed samples.
 * @param fir_data Pointer to struct of type td_block_fir_data_t from which the data samples will be obtained.
 * @param fir_filter Pointer to struct of type td_block_fir_filter_t from which the coefficients will be obtained.
 */
void td_block_fir_compute_ref(
    int32_t samples_out[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data, 
    td_block_fir_filter_t * fir_filter
); 
