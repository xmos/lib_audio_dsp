// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

// #include "dsp/fir.h"

// Normally we would #include "ifft_config.h", but this module has
// no runtime configurable parameters
#include "bump_allocator.h"
#include <xmath/xmath.h>

typedef struct {
    int32_t nfft;
    int32_t* data;
    int32_t exp;
    bfp_s32_t signal;
    bfp_complex_s32_t spectrum;
} ifft_t;

typedef struct
{
    ifft_t* ifft;
    int n_inputs;
    int n_outputs;
    int frame_size;
}ifft_state_t;

typedef struct
{
    int32_t* shared_memory;
    int32_t ifft_len;
}ifft_constants_t;




#define IFFT_STAGE_REQUIRED_MEMORY(N_CH, SAMPLES) \
     + ADSP_BUMP_ALLOCATOR_WORD_N_BYTES(N_CH * sizeof(ifft_t)))

void ifft_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void ifft_process(int32_t **input, int32_t **output, void *app_data_state);

void ifft_control(void *state, module_control_t *control);
