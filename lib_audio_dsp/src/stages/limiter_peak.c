// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/limiter_peak.h"

static inline void limiter_copy_config_to_state(limiter_t *lim_state, int n_inputs, const limiter_peak_config_t *lim_config)
{
    // Same config for all channels
    for(int i=0; i<n_inputs; i++)
    {
        lim_state[i].env_det.attack_alpha = lim_config->attack_alpha;
        lim_state[i].env_det.release_alpha = lim_config->release_alpha;
        lim_state[i].threshold = lim_config->threshold;
    }
}

static inline void limiter_copy_state_to_config(limiter_peak_config_t *lim_config, const limiter_t *lim_state)
{
    // Copy from channel 0 state to the config
    lim_config->attack_alpha = lim_state[0].env_det.attack_alpha;
    lim_config->release_alpha = lim_state[0].env_det.release_alpha;
    lim_config->envelope = lim_state[0].env_det.envelope;
    lim_config->gain = lim_state[0].gain;
    lim_config->threshold = lim_state[0].threshold;
}

void limiter_peak_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    limiter_peak_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];

        int j = 0;
        do {
            *out++ = adsp_limiter_peak(&state->lim[i], *in++);
        } while(++j < state->frame_size);
    } while(++i < state->n_outputs);
}

void limiter_peak_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    xassert(n_inputs == n_outputs && "Limiter should have the same number of inputs and outputs");
    limiter_peak_state_t *state = instance->state;
    limiter_peak_config_t *config = instance->control.config;

    memset(state, 0, sizeof(limiter_peak_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->lim = adsp_bump_allocator_malloc(allocator, state->n_inputs * sizeof(limiter_t));
    memset(state->lim, 0, state->n_inputs * sizeof(limiter_t));

    for(int i=0; i<state->n_inputs; i++)
    {
        state->lim[i].gain = INT32_MAX;
        state->lim[i].env_det.envelope = 0;
    }

    limiter_copy_config_to_state(state->lim, state->n_inputs, config);
}

void limiter_peak_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    limiter_peak_state_t *state = module_state;
    xassert(control != NULL);
    limiter_peak_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        // TODO update only the fields written by the host
        limiter_copy_config_to_state(state->lim, state->n_inputs, config);
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        limiter_copy_state_to_config(config, state->lim);
        control->config_rw_state = config_read_updated;
    }
    else {
        // nothing to do
    }
}
