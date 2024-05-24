// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/biquad.h"
#include "biquad_config.h" // Autogenerated
#include "bump_allocator.h"

#define BIQUAD_STATE_LEN (8)
typedef struct
{
    biquad_config_t config;
    int32_t **filter_states;
    int n_inputs;
    int n_outputs;
    int frame_size;
}biquad_state_t;


#define _BQ_FILTER_MEMORY \
    (BIQUAD_STATE_LEN * sizeof(int32_t))
#define _BQ_ALL_FILTER_MEMORY(N_IN) \
    (ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(_BQ_FILTER_MEMORY) * (N_IN))
#define _BQ_ARR_MEMORY(N_IN) \
    ((N_IN) * sizeof(int32_t*))


#define BIQUAD_STAGE_REQUIRED_MEMORY(N_IN) \
    _BQ_ALL_FILTER_MEMORY(N_IN) + _BQ_ARR_MEMORY(N_IN)

void biquad_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void biquad_process(int32_t **input, int32_t **output, void *app_data_state, int32_t frame_size, int32_t channels);

void biquad_control(void *state, module_control_t *control);
