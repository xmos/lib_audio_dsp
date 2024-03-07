// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/adsp.h"
#include "volume_control_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    volume_control_t * vol_ctl;
    int n_inputs;
    int n_outputs;
    int frame_size;
}volume_control_state_t;

#define VOLUME_CONTROL_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (N_IN * sizeof(volume_control_t))

void volume_control_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void volume_control_process(int32_t **input, int32_t **output, void *app_data_state);

void volume_control_control(void *state, module_control_t *control);

