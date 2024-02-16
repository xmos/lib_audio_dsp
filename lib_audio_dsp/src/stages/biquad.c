// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/biquad.h"

void biquad_process(int32_t **input, int32_t **output, void *app_data_state)
{
    biquad_state_t *state = app_data_state;

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
            *out++ = adsp_biquad((*in++),
                        state->config.filter_coeffs,
                        state->filter_states[i],
                        state->config.left_shift);
        } while (++j < state->frame_size);
    } while (++i < state->n_outputs);
}

module_instance_t* biquad_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config)
{
    module_instance_t *module_instance = malloc(sizeof(module_instance_t));

    biquad_state_t *state = malloc(sizeof(biquad_state_t)); // malloc_from_heap
    biquad_config_t *config = malloc(sizeof(biquad_config_t)); // malloc_from_heap

    memset(state, 0, sizeof(biquad_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->filter_states = malloc(n_inputs * sizeof(int32_t*)); // Allocate memory for the 1D pointers
    for(int i=0; i<n_inputs; i++)
    {
        state->filter_states[i] = DWORD_ALIGNED_MALLOC(BIQUAD_STATE_LEN * sizeof(int32_t));
        memset(state->filter_states[i], 0, BIQUAD_STATE_LEN * sizeof(int32_t));
    }

    xassert(module_config != NULL);

    biquad_config_t *init_config = module_config;
    memcpy(&state->config, init_config, sizeof(biquad_config_t));

    memcpy(config, &state->config, sizeof(biquad_config_t));

    module_instance->state = state;

    // Control stuff
    module_instance->control.config = config;
    module_instance->control.id = id;
    module_instance->control.module_type = e_dsp_stage_biquad;
    module_instance->control.num_control_commands = NUM_CMDS_BIQUAD;
    module_instance->control.config_rw_state = config_none_pending;
    return module_instance;
}

void biquad_control(void *module_state, module_control_t *control)
{
    biquad_state_t *state = module_state;
    biquad_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(biquad_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(biquad_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}
