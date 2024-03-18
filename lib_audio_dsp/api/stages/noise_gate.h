// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "bump_allocator.h"
#include "dsp/drc.h"
#include "noise_gate_config.h" // Autogenerated

typedef struct
{
    noise_gate_t *ng;
    int n_inputs;
    int n_outputs;
    int frame_size;
}noise_gate_state_t;

#define NOISE_GATE_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (N_IN * sizeof(noise_gate_t))

void noise_gate_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void noise_gate_process(int32_t **input, int32_t **output, void *app_data_state);

void noise_gate_control(void *state, module_control_t *control);
