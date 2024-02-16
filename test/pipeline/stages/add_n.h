// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
/// Simple test stage which add a configurable value to the input
///
#pragma once

#include "add_n_config.h" // Autogenerated

typedef struct
{
    add_n_config_t config;
    int n_inputs;
    int n_outputs;
    int frame_size;
}add_n_state_t;

void add_n_init(module_instance_t* instance, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void add_n_process(int32_t **input, int32_t **output, void *app_data_state);

void add_n_control(void *state, module_control_t *control);
