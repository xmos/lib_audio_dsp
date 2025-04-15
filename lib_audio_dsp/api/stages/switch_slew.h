// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "switch_slew_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    switch_slew_config_t config;
    switch_slew_t switch_state;
    int n_inputs;
    int n_outputs;
    int frame_size;
}switch_slew_state_t;

typedef struct
{
    uint32_t fs;
} switch_slew_constants_t;

#define SWITCH_SLEW_STAGE_REQUIRED_MEMORY 0

void switch_slew_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void switch_slew_process(int32_t **input, int32_t **output, void *app_data_state);

void switch_slew_control(void *state, module_control_t *control);

