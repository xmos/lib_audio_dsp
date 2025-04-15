// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "bump_allocator.h"
#include "dsp/drc.h"
#include "noise_suppressor_expander_config.h" // Autogenerated

typedef struct
{
    noise_suppressor_expander_t *nse;
    int n_inputs;
    int n_outputs;
    int frame_size;
}noise_suppressor_expander_state_t;

#define NOISE_SUPPRESSOR_EXPANDER_STAGE_REQUIRED_MEMORY(N_IN) (N_IN * ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(sizeof(noise_suppressor_expander_t)))
#define NOISE_SUPPRESSOR_EXPANDER_STAGE_REQUIRED_MEMORY_SLIM(N_IN) (N_IN * sizeof(noise_suppressor_expander_t))

void noise_suppressor_expander_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void noise_suppressor_expander_process(int32_t **input, int32_t **output, void *app_data_state);

void noise_suppressor_expander_control(void *state, module_control_t *control);
