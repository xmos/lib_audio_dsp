// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#ifndef FD_BLOCK_FIR
#define FD_BLOCK_FIR
#include "xmath/types.h"

typedef struct fd_FIR_data_t {
    bfp_complex_s32_t * data_blocks;
    int32_t * prev_td_data;
    int32_t * overlapping_frame_data;
    uint32_t head_index;

    // These are the three properties
    uint32_t td_block_length; // The length of the block when in a TD form including zero padding
    uint32_t block_count;
    uint32_t frame_advance;

} fd_FIR_data_t;

typedef struct fd_FIR_filter_t {
    bfp_complex_s32_t * coef_blocks;

    // These are the three properties
    uint32_t td_block_length; // The length of the block when in a TD form including zero padding
    uint32_t block_count;
    uint32_t taps_per_block;

} fd_FIR_filter_t;

/*
for a filter and data to be convolveable:
  - filter->block_length == data->block_length
  - filter->block_count <= data->block_count
*/

void fd_block_fir_data_init(fd_FIR_data_t * d, int32_t *data,
    uint32_t frame_advance, uint32_t block_length, uint32_t block_count);

void fd_block_fir_add_data(
    int32_t * samples_in,
    fd_FIR_data_t * fir_data);
    
void fd_block_fir_compute(
    int32_t * samples_out,
    fd_FIR_data_t * fir_data,
    fd_FIR_filter_t * fir_filter);

#endif