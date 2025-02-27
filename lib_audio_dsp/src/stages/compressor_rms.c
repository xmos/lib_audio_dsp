// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/compressor_rms.h"

static inline void compressor_copy_config_to_state(compressor_t *comp_state, int n_inputs, const compressor_rms_config_t *comp_config)
{
    // Same config for all channels
    for(int i=0; i<n_inputs; i++)
    {
        comp_state[i].env_det.attack_alpha = comp_config->attack_alpha;
        comp_state[i].env_det.release_alpha = comp_config->release_alpha;
        comp_state[i].threshold = comp_config->threshold;
	    comp_state[i].slope = comp_config->slope;
    }
}

static inline void compressor_copy_state_to_config(compressor_rms_config_t *comp_config, const compressor_t *comp_state)
{
    // Copy from channel 0 state to the config
    comp_config->attack_alpha = comp_state[0].env_det.attack_alpha;
    comp_config->release_alpha = comp_state[0].env_det.release_alpha;
    comp_config->envelope = comp_state[0].env_det.envelope;
    comp_config->gain = comp_state[0].gain;
    comp_config->threshold = comp_state[0].threshold;
    comp_config->slope = comp_state[0].slope;
}

void compressor_rms_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    compressor_rms_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];

        int j = 0;
        do {
            *out++ = adsp_compressor_rms(&state->comp[i], *in++);
        } while(++j < state->frame_size);
    } while(++i < state->n_outputs);
}

void compressor_rms_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    xassert(n_inputs == n_outputs && "Compressor should have the same number of inputs and outputs");
    compressor_rms_state_t *state = instance->state;
    compressor_rms_config_t *config = instance->control.config;

    memset(state, 0, sizeof(compressor_rms_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->comp = adsp_bump_allocator_malloc(allocator, COMPRESSOR_RMS_STAGE_REQUIRED_MEMORY(state->n_inputs));
    memset(state->comp, 0, COMPRESSOR_RMS_STAGE_REQUIRED_MEMORY(state->n_inputs));

    for(int i=0; i<state->n_inputs; i++)
    {
        state->comp[i].gain = 1 << 30;
        state->comp[i].env_det.envelope = 0;
    }

    compressor_copy_config_to_state(state->comp, state->n_inputs, config);
}

void compressor_rms_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    compressor_rms_state_t *state = module_state;
    xassert(control != NULL);
    compressor_rms_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        // TODO update only the fields written by the host
        compressor_copy_config_to_state(state->comp, state->n_inputs, config);
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        compressor_copy_state_to_config(config, state->comp);
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do
    }
}
