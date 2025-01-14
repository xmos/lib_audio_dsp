// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/mixer.h"



void mixer_process(int32_t **input, int32_t **output, void *app_data_state)
{
    mixer_state_t *state = app_data_state;

    for(int sample_index = 0; sample_index < state->frame_size; ++sample_index) {
        int32_t *out = &output[0][sample_index];
        int64_t acc = 0;
        for(int input_index = 0; input_index < state->n_inputs; ++input_index) {
            acc += adsp_fixed_gain(input[input_index][sample_index], state->config.gain);
        }
        *out = adsp_saturate_32b(acc);
    }
}

void mixer_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    mixer_state_t *state = instance->state;
    mixer_config_t *config = instance->control.config;

    memset(state, 0, sizeof(mixer_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    xassert(n_outputs == 1 && "Mixer should only have one output");

    memcpy(&state->config, config, sizeof(mixer_config_t));
}

void mixer_control(void *module_state, module_control_t *control)
{
    mixer_state_t *state = module_state;
    mixer_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(mixer_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(mixer_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}
