// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/add_n.h"

void add_n_process(int32_t **input, int32_t **output, void *app_data_state)
{
    add_n_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];
        int j = 0;
        do
        {
            out[j] = in[j] + state->config.n;
        } while (++j < state->frame_size);
    } while (++i < state->n_outputs);
}

void add_n_init(module_instance_t* instance, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    add_n_state_t *state = instance->state;
    add_n_config_t *config = instance->control.config;

    memset(state, 0, sizeof(add_n_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    memcpy(&state->config, config, sizeof(add_n_config_t));
}

void add_n_control(void *module_state, module_control_t *control)
{
    add_n_state_t *state = module_state;
    add_n_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(add_n_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(add_n_config_t));
        control->config_rw_state = config_read_updated;
    }
}
