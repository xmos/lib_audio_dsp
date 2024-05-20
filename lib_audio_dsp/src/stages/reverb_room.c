// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/reverb_room.h"

void reverb_room_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size)
{
    xassert(n_inputs == n_outputs && "Reverb should have the same number of inputs and outputs");
    reverb_room_state_t *state = instance->state;
    reverb_room_config_t *config = instance->control.config;

    memset(state, 0, sizeof(reverb_room_state_t));

    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    float fs = config->sampling_freq;
    float max_room_size = config->max_room_size;

    float room_size = config->room_size;
    int32_t feedback = config->feedback;
    int32_t damping = config->damping;

    xassert(n_inputs == 1); // Currently support only 1 channel reverb

    uint8_t *reverb_heap = adsp_bump_allocator_malloc(allocator, REVERB_ROOM_STAGE_REQUIRED_MEMORY(fs, max_room_size));
    memset(reverb_heap, 0, REVERB_ROOM_STAGE_REQUIRED_MEMORY(fs, max_room_size));

    state->rv.pre_gain = config->pregain;
    state->rv.wet_gain = config->wet_gain;
    state->rv.dry_gain = config->dry_gain;

    adsp_reverb_room_init_filters(&state->rv, fs, max_room_size, feedback, damping, reverb_heap);
    adsp_reverb_room_set_room_size(&state->rv, room_size);
}

void reverb_room_process(int32_t **input, int32_t **output, void *app_data_state)
{
    reverb_room_state_t *state = app_data_state;
    int32_t *in = input[0];
    int32_t *out = output[0];
    int j = 0;
    do
    {
        *out++ = adsp_reverb_room(&state->rv, (*in++));
    } while (++j < state->frame_size);
}

void reverb_room_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    reverb_room_state_t *state = module_state;
    xassert(control != NULL);
    reverb_room_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        state->rv.pre_gain = config->pregain;
        state->rv.wet_gain = config->wet_gain;
        state->rv.dry_gain = config->dry_gain;
        if (config->room_size != state->rv.room_size) {
            adsp_reverb_room_set_room_size(&state->rv, config->room_size);
        }
        for (unsigned i = 0; i < ADSP_RVR_N_COMBS; i ++) {
            state->rv.combs[i].feedback = config->feedback;
            state->rv.combs[i].damp_1 = config->damping;
            // damping is always at least 1
            state->rv.combs[i].damp_2 = (uint32_t)(1<<31) - config->damping;
        }
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        // none of these should be changed during the reverb execution,
        // so don't really need to update the config,
        // leaving it here in case something goes really wrong,
        // so there's a way to debug
        config->pregain = state->rv.pre_gain;
        config->wet_gain = state->rv.wet_gain;
        config->dry_gain = state->rv.dry_gain;
        config->room_size = state->rv.room_size;
        config->feedback = state->rv.combs[0].feedback;
        config->damping = state->rv.combs[0].damp_1;
        control->config_rw_state = config_read_updated;
    }
    else {
        // nothing to do
    }

}
