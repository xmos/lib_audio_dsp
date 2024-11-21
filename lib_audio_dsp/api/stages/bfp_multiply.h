// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "bump_allocator.h"
#include <xmath/xmath.h>
#include "dsp/defines.h"

typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
    int nfft;
    int exp;
}bfp_multiply_state_t;

typedef struct
{
    int32_t nfft;
    int32_t exp;
}bfp_multiply_constants_t;


#define BFP_MULTIPLY_STAGE_REQUIRED_MEMORY 0

void bfp_multiply_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void bfp_multiply_process(complex_spectrum_t **input, complex_spectrum_t **output, void *app_data_state);


