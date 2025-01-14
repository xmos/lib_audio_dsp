// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "bump_allocator.h"

typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
}subtractor_state_t;


#define SUBTRACTOR_STAGE_REQUIRED_MEMORY 0

void subtractor_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void subtractor_process(int32_t **input, int32_t **output, void *app_data_state);


