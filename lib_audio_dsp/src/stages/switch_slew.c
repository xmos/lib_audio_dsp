// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/switch_slew.h"
#include <stdio.h>
#include "print.h"

void switch_slew_process(int32_t **input, int32_t **output, void *app_data_state)
{
    switch_slew_state_t *state = app_data_state;
    int32_t *out = output[0];

    for (int i = 0; i < state->frame_size; i++){
        // put the samples into one array
        int32_t samples[16];
        for (int j = 0; j < state -> n_inputs; j++){
            samples[j] = input[j][i];
        }

        out[i] = adsp_switch_slew(&(state->switch_state),
                                  samples);
    }
}

void switch_slew_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    switch_slew_state_t *state = instance->state;
    switch_slew_config_t *config = instance->control.config;
    switch_slew_constants_t *constants = instance->constants;

    memset(state, 0, sizeof(switch_slew_state_t));
    state->n_inputs = n_inputs;
    xassert(n_inputs <= 16 && "switch_slew should have less than 16 inputs");
    state->frame_size = frame_size;
    xassert(n_outputs == 1 && "switch_slew should only have one output");
    state->n_outputs = n_outputs;
    state->switch_state = adsp_switch_slew_init(constants->fs, config->position);
    memcpy(&state->config, config, sizeof(switch_slew_config_t));
}

void switch_slew_control(void *module_state, module_control_t *control)
{
    switch_slew_state_t *state = module_state;
    switch_slew_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(switch_slew_config_t));
        control->config_rw_state = config_none_pending;
        adsp_switch_slew_move(&(state->switch_state), config->position);
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(switch_slew_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}

