// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#ifndef FD_BLOCK_FIR
#define FD_BLOCK_FIR
#include "xmath/types.h"

typedef struct fd_fir_data_t {
    bfp_complex_s32_t * data_blocks;
    int32_t * prev_td_data;
    int32_t * overlapping_frame_data;
    uint32_t head_index;

    // These are the three properties
    uint32_t td_block_length; // The length of the block when in a TD form including zero padding
    uint32_t block_count;
    uint32_t frame_advance;

} fd_fir_data_t;

typedef struct fd_fir_filter_t {
    bfp_complex_s32_t * coef_blocks;

    // These are the three properties
    uint32_t td_block_length; // The length of the block when in a TD form including zero padding
    uint32_t block_count;
    uint32_t taps_per_block;

} fd_fir_filter_t;

/*
For a filter and data to be convolve-able:
  - filter->block_length == data->block_length
  - filter->block_count <= data->block_count
*/


/*
Function to initialise the struct that manages the data, rather than coefficients, for a 
frequency domain convolution. The filter generator should be run first resulting in a header 
that defines the parameters for this function. For example, running the generator with --name={NAME}
would 
    fir_data - Struct of type fd_fir_data_t
    data - an amount of memory to be used by the struct in order to hold a history of the samples. The 
           define {NAME}_DATA_BUFFER_ELEMENTS specifies exactly the number of int32_t elements to 
           allocate for the filter {NAME} to correctly function.
    frame_advance - The number of samples contained in each frame, i.e. the samples count between updates.
           This should be initialised to {NAME}_FRAME_ADVANCE.
    block_length - The length of the processing block, independent to the frame_advance. Must be a power 
           of two. This should be initialised to {NAME}_TD_BLOCK_LENGTH.
    block_count - The count of blocks required to implement the filter. This should be initialised to 
           {NAME}_BLOCK_COUNT).
*/
void fd_block_fir_data_init(
    fd_fir_data_t * fir_data, 
    int32_t *data,
    uint32_t frame_advance, 
    uint32_t block_length, 
    uint32_t block_count);

/*
Function to add samples to the FIR data structure.
    samples_in - array of int32_t samples of length expected to be fir_data->frame_advance.
    fir_data - Struct of type fd_fir_data_t to which the samples will be added.
*/
void fd_block_fir_add_data(
    int32_t * samples_in,
    fd_fir_data_t * fir_data);
    
/*
Function to compute the convolution between fir_data and fir_filter.
    samples_out - Array of length fir_data->td_block_length, which will be used to return the 
        processed samples. The samples will be returned from element 0 for 
        (fir_data-td_block_length + 1 - fir_filter->taps_per_block) elements. The remaining 
        samples of the array are used as scratch for the processing to be in-place.
    fir_data - Struct of type fd_fir_data_t to which the data samples will be used.
    fir_filter - Struct of type fd_fir_filter_t to which the coefficient samples will be used.
*/
void fd_block_fir_compute(
    int32_t * samples_out,
    fd_fir_data_t * fir_data,
    fd_fir_filter_t * fir_filter);

#endif