// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/envelope_detector_peak.h"


void envelope_detector_peak_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    envelope_detector_peak_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];

        int j = 0;
        do {
            adsp_env_detector_peak(&state->env_det[i], *in++);
        } while(++j < state->frame_size);
    } while(++i < state->n_inputs);
}

void envelope_detector_peak_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    xassert(n_outputs == 0 && "Envelope detector shold not have any outputs");
    envelope_detector_peak_state_t *state = instance->state;
    envelope_detector_peak_config_t *config = instance->control.config;

    memset(state, 0, sizeof(envelope_detector_peak_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->env_det = adsp_bump_allocator_malloc(allocator, ENVELOPE_DETECTOR_PEAK_STAGE_REQUIRED_MEMORY(state->n_inputs));

    for(int i=0; i<state->n_inputs; i++)
    {
        state->env_det[i].envelope = 0;
        state->env_det[i].attack_alpha = config->attack_alpha;
        state->env_det[i].release_alpha = config->release_alpha;
    }
}

void envelope_detector_peak_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    envelope_detector_peak_state_t *state = module_state;
    xassert(control != NULL);
    envelope_detector_peak_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        // TODO update only the fields written by the host
        for(int i=0; i<state->n_inputs; i++)
        {
            state->env_det[i].attack_alpha = config->attack_alpha;
            state->env_det[i].release_alpha = config->release_alpha;
        }
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        config->envelope = state->env_det[0].envelope;
        config->attack_alpha = state->env_det[0].attack_alpha;
        config->release_alpha = state->env_det[0].release_alpha;
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do
    }
}
