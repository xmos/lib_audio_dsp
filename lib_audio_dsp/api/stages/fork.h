// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "fork_config.h" // Autogenerated

typedef struct
{
    fork_config_t config;
    int n_inputs;
    int n_outputs;
    int frame_size;
    int n_forks;  // n_outputs / n_inputs
}fork_state_t;

void fork_init(module_instance_t* instance, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void fork_process(int32_t **input, int32_t **output, void *app_data_state);

void fork_control(void *state, module_control_t *control);
