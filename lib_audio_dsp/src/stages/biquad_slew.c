// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/biquad_slew.h"
#include "dsp/biquad.h"

void biquad_slew_process(int32_t **input, int32_t **output, void *app_data_state)
{
    biquad_slew_state_t *state = app_data_state;

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
            *out++ = adsp_biquad_slew((*in++),
                        state->coeffs[i],
                        state->config.filter_coeffs,
                        state->filter_states[i],
                        state->config.left_shift,
                        state->config.slew_shift);
        } while (++j < state->frame_size);
    } while (++i < state->n_outputs);
}

void biquad_slew_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size)
{
    xassert(n_inputs == n_outputs && "Biquad should have the same number of inputs and outputs");
    biquad_slew_state_t *state = instance->state;
    biquad_slew_config_t *config = instance->control.config;

    memset(state, 0, sizeof(biquad_slew_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->filter_states = adsp_bump_allocator_malloc(allocator, _BQ_SLEW_ARR_MEMORY(n_inputs)); // Allocate memory for the 1D pointers
    state->coeffs = adsp_bump_allocator_malloc(allocator, _BQ_SLEW_ARR_MEMORY(n_inputs)); // Allocate memory for the 1D pointers
    for(int i=0; i<n_inputs; i++)
    {
        state->filter_states[i] = ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, _BQ_SLEW_FILTER_MEMORY);
        memset(state->filter_states[i], 0, _BQ_SLEW_FILTER_MEMORY);

        // initialise the filter coeffs to the starting values
        state->coeffs[i] = ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, _BQ_SLEW_FILTER_MEMORY);
        memcpy(state->coeffs[i], config->filter_coeffs, 5*sizeof(int32_t));
    }

    // copy default config
    memcpy(&state->config, config, sizeof(biquad_slew_config_t));

}

void biquad_slew_control(void *module_state, module_control_t *control)
{
    biquad_slew_state_t *state = module_state;
    biquad_slew_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(biquad_slew_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(biquad_slew_config_t));
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}
