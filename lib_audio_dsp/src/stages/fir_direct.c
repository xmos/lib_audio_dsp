// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/fir_direct.h"

void fir_direct_process(int32_t ** input, int32_t ** output, void * app_data_state)
{
    xassert(app_data_state != NULL);
    fir_direct_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];
        filter_fir_s32_t* this_filter = &(state->fir_direct[i].filter);

        int j = 0;
        do {
            *out++ = filter_fir_s32(this_filter, *in++);
        } while(++j < state->frame_size);
    } while(++i < state->n_outputs);
}

void fir_direct_init(module_instance_t* instance,
                adsp_bump_allocator_t* allocator,
                uint8_t id,
                int n_inputs,
                int n_outputs,
                int frame_size)
{
    xassert(n_inputs == n_outputs && "fir_direct should have the same number of inputs and outputs");
    fir_direct_state_t *state = instance->state;
    fir_direct_config_t *config = instance->control.config;

    memset(state, 0, sizeof(fir_direct_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    state->n_taps = config->n_taps;
    state->max_taps = config->n_taps;
    state->fir_direct = ADSP_BUMP_ALLOCATOR_WORD_ALLIGNED_MALLOC(allocator, n_inputs * sizeof(fir_direct_t));
    state->coeffs = (int32_t *)ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(config->n_taps));
    memcpy(state->coeffs, config->coeffs, FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(config->n_taps));
    state->shift = config->shift;


    for(int i = 0; i < n_inputs; i++)
    {
        state->fir_direct[i].buffer = (int32_t *)ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(config->n_taps));
        memset(state->fir_direct[i].buffer, 0, FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(config->n_taps));
        filter_fir_s32_init(&(state->fir_direct[i].filter), state->fir_direct[i].buffer, state->n_taps, state->coeffs, state->shift);
    }
}

void fir_direct_control(void *state, module_control_t *control)
{
    xassert(state != NULL);
    fir_direct_state_t *fir_direct_state = state;
    xassert(control != NULL);
    fir_direct_config_t *fir_direct_config = control->config;

    xassert(fir_direct_config->n_taps <= fir_direct_state->max_taps);

    if(control->config_rw_state == config_write_pending) {

        fir_direct_state->n_taps = fir_direct_config->n_taps;
        fir_direct_state->shift = fir_direct_config->shift;

        // coeffs and buffer are already allocated, so just copy data and reinit fir_s32 struct
        memcpy(fir_direct_state->coeffs, fir_direct_config->coeffs, FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(fir_direct_config->n_taps));

        for(int i = 0; i < fir_direct_state->n_inputs; i++)
        {
            filter_fir_s32_init(&(fir_direct_state->fir_direct[i].filter), fir_direct_state->fir_direct[i].buffer, fir_direct_state->n_taps, fir_direct_state->coeffs, fir_direct_state->shift);
        }
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending) {
        fir_direct_config->n_taps = fir_direct_state->n_taps;
        fir_direct_config->shift = fir_direct_state->shift;
        memcpy(fir_direct_config->coeffs, fir_direct_state->coeffs, FIR_DIRECT_DSP_REQUIRED_MEMORY_SAMPLES(fir_direct_state->n_taps));

        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do
    }
}
