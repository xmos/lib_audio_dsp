// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/fir.h"

// Normally we would #include "fft_config.h", but this module has
// no runtime configurable parameters
#include "bump_allocator.h"
#include <xmath/xmath.h>

typedef struct {
} fft_config_t;


typedef struct {
    int32_t nfft;
    int32_t* data;
    int32_t exp;
    bfp_s32_t signal;
    bfp_complex_s32_t spectrum;
} fft_t;

typedef struct
{
    fft_t* fft;
    int n_inputs;
    int n_outputs;
    int frame_size;
}fft_state_t;

typedef struct
{
    int32_t nfft;
    int32_t exp;
}fft_constants_t;




#define FFT_STAGE_REQUIRED_MEMORY(N_CH) \
     + ADSP_BUMP_ALLOCATOR_WORD_N_BYTES(N_CH * sizeof(fft_t))

void fft_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void fft_process(int32_t **input, int32_t **output, void *app_data_state);

void fft_control(void *state, module_control_t *control);
