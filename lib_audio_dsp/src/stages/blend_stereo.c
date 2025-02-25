// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/blend_stereo.h"
#include <stdio.h>
#include "print.h"
#include "control/signal_chain.h"

void blend_stereo_process(int32_t **input, int32_t **output, void *app_data_state)
{
    blend_stereo_state_t *state = app_data_state;

    int32_t *in1_L = input[0];
    int32_t *in1_R = input[1];
    int32_t *in2_L = input[2];
    int32_t *in2_R = input[3];
    int32_t *out_L = output[0];
    int32_t *out_R = output[1];

    int j = 0;
    do
    {
        *out_L++ = adsp_blend(*in1_L++, *in2_L++, state->gains[0], state->gains[1], 31);
        *out_R++ = adsp_blend(*in1_R++, *in2_R++, state->gains[0], state->gains[1], 31);
    } while (++j < state->frame_size);
}


void blend_stereo_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    blend_stereo_state_t *state = instance->state;
    blend_stereo_config_t *config = instance->control.config;

    memset(state, 0, sizeof(blend_stereo_state_t));
    state->n_inputs = n_inputs;
    xassert(n_inputs == 4 && "Stereo blend must have 2 stereo inputs");

    state->frame_size = frame_size;
    xassert(n_outputs == 2 && "Stereo blend should only have two outputs");
    state->n_outputs = n_outputs;

    memcpy(&state->config, config, sizeof(blend_stereo_config_t));
    adsp_blend_mix(state->gains, config->mix);
}

void blend_stereo_control(void *module_state, module_control_t *control)
{
    blend_stereo_state_t *state = module_state;
    blend_stereo_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(blend_stereo_config_t));
        control->config_rw_state = config_none_pending;
        adsp_blend_mix(state->gains, config->mix);
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(blend_stereo_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}

