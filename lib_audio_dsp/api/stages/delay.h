// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/signal_chain.h"
#include "delay_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    delay_t *delay;
    int n_inputs;
    int n_outputs;
    int frame_size;
}delay_state_t;

#define DELAY_STAGE_REQUIRED_MEMORY(N_CH, SAMPLES) (N_CH * (DELAY_DSP_REQUIRED_MEMORY_SAMPLES(SAMPLES) + sizeof(delay_t)))

void delay_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void delay_process(int32_t **input, int32_t **output, void *app_data_state);

void delay_control(void *state, module_control_t *control);
