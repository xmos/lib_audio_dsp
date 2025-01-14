// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "bump_allocator.h"

typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
}bypass_state_t;

#define BYPASS_STAGE_REQUIRED_MEMORY 0

void bypass_init(module_instance_t* module_instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void bypass_process(int32_t **input, int32_t **output, void *app_data_state);
