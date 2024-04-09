// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/noise_suppressor.h"
#include "dsp/adsp.h"

static inline void ns_copy_config_to_state(noise_suppressor_t *ns_state, int n_inputs, const noise_suppressor_config_t *ns_config)
{
    // Avoid division by 0
    int32_t th = (!ns_config->threshold) ? 1 : ns_config->threshold;
    int32_t condition = (ns_state[0].threshold != th); // threshold is the same in all channels
    // Compute the inverse of the threshold only if the threshold has changed
    int64_t inv_th = (condition) ? INT64_MAX / th : 0; // else doesn't matter here
    // Same config for all channels
    for(int i=0; i<n_inputs; i++)
    {
        ns_state[i].env_det.attack_alpha = ns_config->attack_alpha;
        ns_state[i].env_det.release_alpha = ns_config->release_alpha;
        ns_state[i].slope = ns_config->slope;
        // Change the inverse of the threshold only if the threshold has changed
        if (condition)
        {
            ns_state[i].threshold = th;
            ns_state[i].inv_threshold = inv_th;
        }
    }
}

static inline void ns_copy_state_to_config(noise_suppressor_config_t *ns_config, const noise_suppressor_t *ns_state)
{
    // Copy from channel 0 state to the config
    ns_config->attack_alpha = ns_state[0].env_det.attack_alpha;
    ns_config->release_alpha = ns_state[0].env_det.release_alpha;
    ns_config->envelope = ns_state[0].env_det.envelope;
    ns_config->gain = ns_state[0].gain;
    ns_config->threshold = ns_state[0].threshold;
    ns_config->slope = ns_state[0].slope;
}

void noise_suppressor_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    noise_suppressor_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];

        int j = 0;
        do {
            *out++ = adsp_noise_suppressor(&state->ns[i], *in++);
        } while(++j < state->frame_size);
    } while(++i < state->n_outputs);
}

void noise_suppressor_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    xassert(n_inputs == n_outputs && "Noise suppressor should have the same number of inputs and outputs");
    noise_suppressor_state_t *state = instance->state;
    noise_suppressor_config_t *config = instance->control.config;

    memset(state, 0, sizeof(noise_suppressor_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    state->ns = ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, state->n_inputs * sizeof(noise_suppressor_t));
    memset(state->ns, 0, state->n_inputs * sizeof(noise_suppressor_t));

    for(int i=0; i<state->n_inputs; i++)
    {
        state->ns[i].gain = INT32_MAX;
        state->ns[i].env_det.envelope = 1 << (-SIG_EXP);
        // Avoid division by zero
        if (!state->ns[i].threshold) state->ns[i].threshold = 1;
        state->ns[i].inv_threshold = INT64_MAX / state->ns[i].threshold;
    }

    ns_copy_config_to_state(state->ns, state->n_inputs, config);
}

void noise_suppressor_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    noise_suppressor_state_t *state = module_state;
    xassert(control != NULL);
    noise_suppressor_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        // TODO update only the fields written by the host
        ns_copy_config_to_state(state->ns, state->n_inputs, config);
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        ns_copy_state_to_config(config, state->ns);
        control->config_rw_state = config_read_updated;
    }
    else {
        // nothing to do
    }
}
