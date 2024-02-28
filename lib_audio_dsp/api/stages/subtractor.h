// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "subtractor_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    subtractor_config_t config;
    int n_inputs;
    int n_outputs;
    int frame_size;
    int32_t inputs_rearranged[2];
}subtractor_state_t;

#define _SUB_ARR_MEMORY(N_IN) \
    ((N_IN) * sizeof(int32_t*))

#define SUBTRACTOR_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (_SUB_ARR_MEMORY(N_IN))

void subtractor_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void subtractor_process(int32_t **input, int32_t **output, void *app_data_state);

void subtractor_control(void *state, module_control_t *control);

