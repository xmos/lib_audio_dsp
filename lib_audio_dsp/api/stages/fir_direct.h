// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/fir.h"

// Normally we would #include "fir_direct_config.h", but this module has
// no runtime configurable parameters
#include "bump_allocator.h"

typedef struct
{
    fir_direct_t *fir_direct;
    int n_inputs;
    int n_outputs;
    int frame_size;
}fir_direct_state_t;

typedef struct
{
    int32_t n_taps;
    int32_t shift;
    int32_t* coeffs;
}fir_direct_constants_t;




#define FIR_DIRECT_STAGE_REQUIRED_MEMORY(N_CH, SAMPLES) \
    (((N_CH) * ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(SAMPLES))) \
     + ADSP_BUMP_ALLOCATOR_WORD_N_BYTES(N_CH * sizeof(fir_direct_t)))

void fir_direct_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void fir_direct_process(int32_t **input, int32_t **output, void *app_data_state);

void fir_direct_control(void *state, module_control_t *control);
