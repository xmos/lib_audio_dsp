// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once


#include "bump_allocator.h"
#include "dsp/graphic_eq.h"
#include "graphic_eq_10b_config.h" // Autogenerated

#define GEQ10_STATE_LEN (160)

typedef struct
{
    graphic_eq_10b_config_t config;
    int32_t **filter_states;
    q2_30 *filter_coeffs;
    int n_inputs;
    int n_outputs;
    int frame_size;
}graphic_eq_10b_state_t;

typedef struct
{
    q2_30 *coeffs;
    // uint32_t sampling_freq;
}graphic_eq_10b_constants_t;

#define _GEQ10_FILTER_MEMORY \
    (GEQ10_STATE_LEN * sizeof(int32_t))
#define _GEQ10_ALL_FILTER_MEMORY(N_IN) \
    (ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(_GEQ10_FILTER_MEMORY) * (N_IN))
#define _GEQ10_ARR_MEMORY(N_IN) \
    ((N_IN) * sizeof(int32_t*))


#define GRAPHIC_EQ_10B_STAGE_REQUIRED_MEMORY(N_IN) \
    (_GEQ10_ALL_FILTER_MEMORY(N_IN) + _GEQ10_ARR_MEMORY(N_IN))

void graphic_eq_10b_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void graphic_eq_10b_process(int32_t **input, int32_t **output, void *app_data_state);

void graphic_eq_10b_control(void *state, module_control_t *control);
