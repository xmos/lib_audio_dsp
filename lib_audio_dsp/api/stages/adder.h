// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "bump_allocator.h"

typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
}adder_state_t;



#define ADDER_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (0)

void adder_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void adder_process(int32_t **input, int32_t **output, void *app_data_state);


