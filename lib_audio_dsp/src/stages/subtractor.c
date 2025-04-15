// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/subtractor.h"

void subtractor_process(int32_t **input, int32_t **output, void *app_data_state)
{
    subtractor_state_t *state = app_data_state;

    // we have to shuffle the data from [chans, samples] to [samples, chans]
    for(int sample_index = 0; sample_index < state->frame_size; ++sample_index) {
        int32_t *out = &output[0][sample_index];

        *out = adsp_subtractor(input[0][sample_index], input[1][sample_index]);
    }
}

void subtractor_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    subtractor_state_t *state = instance->state;

    memset(state, 0, sizeof(subtractor_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    xassert(n_outputs == 1 && "Subtractor should only have one output");
    xassert(n_inputs == 2 && "Subtractor should only have two inputs");
}


