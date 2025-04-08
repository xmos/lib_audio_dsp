// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/compressor_sidechain.h"

static inline void compressor_copy_config_to_state(compressor_t *comp_state, const compressor_sidechain_config_t *comp_config)
{
        comp_state->env_det.attack_alpha = comp_config->attack_alpha;
        comp_state->env_det.release_alpha = comp_config->release_alpha;
        comp_state->threshold = comp_config->threshold;
        comp_state->slope = comp_config->slope;
}

static inline void compressor_copy_state_to_config(compressor_sidechain_config_t *comp_config, const compressor_t *comp_state)
{
    // Copy from channel 0 state to the config
    comp_config->attack_alpha = comp_state->env_det.attack_alpha;
    comp_config->release_alpha = comp_state->env_det.release_alpha;
    comp_config->envelope = comp_state->env_det.envelope;
    comp_config->gain = comp_state->gain;
    comp_config->threshold = comp_state->threshold;
    comp_config->slope = comp_state->slope;
}

void compressor_sidechain_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    compressor_sidechain_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int32_t *in = input[0];
    int32_t *detect = input[1];
    int32_t *out = output[0];

    int j = 0;
    do {
        *out++ = adsp_compressor_rms_sidechain(&state->comp, *in++, *detect++);
    } while(++j < state->frame_size);
}

void compressor_sidechain_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    xassert(n_inputs == 2 && "Sidechain compressor should have the 2 inputs");
    xassert(n_outputs == 1 && "Sidechain compressor should have 1 output");
    compressor_sidechain_state_t *state = instance->state;
    compressor_sidechain_config_t *config = instance->control.config;

    memset(state, 0, sizeof(compressor_sidechain_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    memset(&state->comp, 0, sizeof(compressor_t));

    state->comp.gain = INT32_MAX;
    state->comp.env_det.envelope = 0;

    compressor_copy_config_to_state(&state->comp, config);
}

void compressor_sidechain_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    compressor_sidechain_state_t *state = module_state;
    xassert(control != NULL);
    compressor_sidechain_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        // TODO update only the fields written by the host
        compressor_copy_config_to_state(&state->comp, config);
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        compressor_copy_state_to_config(config, &state->comp);
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do
    }
}
