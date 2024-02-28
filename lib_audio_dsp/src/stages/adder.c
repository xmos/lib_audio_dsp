// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/adder.h"

void adder_process(int32_t **input, int32_t **output, void *app_data_state)
{
    adder_state_t *state = app_data_state;

    // we have to shuffle the data from [chans, samples] to [samples, chans]
    for(int sample_index = 0; sample_index < state->frame_size; ++sample_index) {
        int32_t *out = &output[0][sample_index];
        int64_t acc = 0;
        for(int input_index = 0; input_index < state->n_inputs; ++input_index) {
            acc += input[input_index][sample_index];
        }
        *out = adsp_saturate_32b(acc);
    }
}

void adder_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    adder_state_t *state = instance->state;
    adder_config_t *config = instance->control.config;

    memset(state, 0, sizeof(adder_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    xassert(n_outputs == 1); // should only have 1 output

    memcpy(&state->config, config, sizeof(adder_config_t));
}

void adder_control(void *module_state, module_control_t *control)
{
    // nothing to do
}

