// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/volume_control.h"



void volume_control_process(int32_t **input, int32_t **output, void *app_data_state)
{
    volume_control_state_t *state = app_data_state;

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
            *out++ = adsp_fixed_gain((*in++), state->config.gain);
        } while (++j < state->frame_size);
    } while (++i < state->n_outputs);

}

void volume_control_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    volume_control_state_t *state = instance->state;
    volume_control_config_t *config = instance->control.config;

    memset(state, 0, sizeof(volume_control_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    state->gain = config->gain;
    xassert(n_outputs == n_inputs);

    memcpy(&state->config, config, sizeof(volume_control_config_t));
}

void volume_control_control(void *module_state, module_control_t *control)
{
    volume_control_state_t *state = module_state;
    volume_control_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(volume_control_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(volume_control_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }}

