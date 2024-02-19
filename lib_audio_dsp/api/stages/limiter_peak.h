// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "bump_allocator.h"
#include "dsp/drc.h"
#include "limiter_peak_config.h" // Autogenerated

typedef struct
{
    limiter_t *lim;
    int n_inputs;
    int n_outputs;
    int frame_size;
}limiter_peak_state_t;

#define LIMITER_PEAK_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (N_IN * sizeof(limiter_t))

void limiter_peak_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void limiter_peak_process(int32_t **input, int32_t **output, void *app_data_state);

void limiter_peak_control(void *state, module_control_t *control);
