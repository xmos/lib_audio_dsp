// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/biquad.h"
#include "biquad_slew_config.h" // Autogenerated
#include "bump_allocator.h"

#define BIQUAD_SLEW_STATE_LEN (8)
typedef struct
{
    biquad_slew_config_t config;
    int32_t **filter_states;
    int32_t **coeffs;
    int n_inputs;
    int n_outputs;
    int frame_size;
    left_shift_t  **remaining_shifts;
    left_shift_t **b_shift;
}biquad_slew_state_t;


#define _BQ_SLEW_FILTER_MEMORY \
    (BIQUAD_SLEW_STATE_LEN * sizeof(int32_t))
#define _BQ_SLEW_ALL_FILTER_MEMORY(N_IN) \
    (ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(_BQ_SLEW_FILTER_MEMORY) * (N_IN))
#define _BQ_SLEW_ARR_MEMORY(N_IN) \
    ((N_IN) * sizeof(int32_t*))


#define BIQUAD_SLEW_STAGE_REQUIRED_MEMORY(N_IN) \
    2*_BQ_SLEW_ALL_FILTER_MEMORY(N_IN) + 2*_BQ_SLEW_ARR_MEMORY(N_IN)

void biquad_slew_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void biquad_slew_process(int32_t **input, int32_t **output, void *app_data_state);

void biquad_slew_control(void *state, module_control_t *control);
