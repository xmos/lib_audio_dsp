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
    xassert(app_data_state != NULL);
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
            *out++ = adsp_volume_control(&state->vol_ctl[i], *in++);
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
    xassert(n_outputs == n_inputs && "Volume control should have the same number of inputs and outputs");

    state->vol_ctl = adsp_bump_allocator_malloc(allocator, VOLUME_CONTROL_STAGE_REQUIRED_MEMORY(state->n_inputs));

    for(int i=0; i<state->n_inputs; i++)
    {
        state->vol_ctl[i].gain = config->user_gain;
        state->vol_ctl[i].user_gain = config->user_gain;
        state->vol_ctl[i].target_gain = config->user_gain;
        state->vol_ctl[i].slew_shift = config->slew_shift;
        state->vol_ctl[i].mute_state = 0;
    }
}

void volume_control_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    volume_control_state_t *state = module_state;
    xassert(control != NULL);
    volume_control_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        for (unsigned i=0; i < state->n_inputs; i++) {
            adsp_volume_control_set_gain(&state->vol_ctl[i], config->user_gain);
            state->vol_ctl[i].slew_shift = config->slew_shift;
            if ((state->vol_ctl[i].mute_state != config->mute_state) && (config->mute_state)) {
                adsp_volume_control_mute(&state->vol_ctl[i]);
            } else if ((state->vol_ctl[i].mute_state != config->mute_state) && (!config->mute_state)) {
                adsp_volume_control_unmute(&state->vol_ctl[i]);
            }
        }
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        config->user_gain = state->vol_ctl[0].user_gain;
        config->gain = state->vol_ctl[0].gain;
        config->slew_shift = state->vol_ctl[0].slew_shift;
        config->mute_state = state->vol_ctl[0].mute_state;

        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}

