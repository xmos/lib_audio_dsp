// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/graphic_eq_10b.h"

void graphic_eq_10b_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    graphic_eq_10b_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];

        int j = 0;
        do {
            *out++ = adsp_graphic_eq_10b((*in++),
                        state->config.gains,
                        state->filter_coeffs,
                        state->filter_states[i]);
        } while(++j < state->frame_size);
    } while(++i < state->n_outputs);
}

void graphic_eq_10b_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    xassert(n_inputs == n_outputs && "Cascaded biuqads should have the same number of inputs and outputs");
    graphic_eq_10b_state_t *state = instance->state;
    graphic_eq_10b_config_t *config = instance->control.config;
    graphic_eq_10b_constants_t *constants = instance->constants;

    memset(state, 0, sizeof(graphic_eq_10b_state_t));

    state->n_inputs = n_inputs;

    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    state->filter_coeffs = constants->coeffs;

    state->filter_states = adsp_bump_allocator_malloc(allocator, _GEQ10_ARR_MEMORY(n_inputs)); // Allocate memory for the 1D pointers
    for(int i=0; i<n_inputs; i++)
    {
        state->filter_states[i] = ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, _GEQ10_FILTER_MEMORY);
        memset(state->filter_states[i], 0, GEQ10_STATE_LEN * sizeof(int32_t));
    }

    memcpy(&state->config, config, sizeof(graphic_eq_10b_config_t));
}

void graphic_eq_10b_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    graphic_eq_10b_state_t *state = module_state;
    xassert(control != NULL);
    graphic_eq_10b_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(graphic_eq_10b_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(graphic_eq_10b_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}
