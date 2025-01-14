// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "fixed_gain_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    fixed_gain_config_t config;
    int n_inputs;
    int n_outputs;
    int frame_size;
}fixed_gain_state_t;

#define FIXED_GAIN_STAGE_REQUIRED_MEMORY 0

void fixed_gain_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void fixed_gain_process(int32_t **input, int32_t **output, void *app_data_state);

void fixed_gain_control(void *state, module_control_t *control);

