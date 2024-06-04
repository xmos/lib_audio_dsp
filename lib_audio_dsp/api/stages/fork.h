// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "bump_allocator.h"

typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
    int n_forks;  // n_outputs / n_inputs
}fork_state_t;


#define FORK_STAGE_REQUIRED_MEMORY 0

void fork_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void fork_process(int32_t **input, int32_t **output, void *app_data_state);

