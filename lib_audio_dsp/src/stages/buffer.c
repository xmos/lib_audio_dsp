// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/buffer.h"

void buffer_process(int32_t ** input, int32_t *** output, void * app_data_state)
{
    xassert(app_data_state != NULL);
    buffer_state_t *state = app_data_state;

    int32_t* d = state->buffer[0].buffer_data;
    int32_t overlap = state->buffer[0].buffer_len - state->frame_size;

    // roll buffer
    printf("rolling buffer by %d\n", state->frame_size);
    memcpy(d, &d[state->frame_size], overlap*sizeof(int32_t));

    // add new samples
    int32_t* samples_in = input[0];
    memcpy(d + state->frame_size, samples_in, state->frame_size*sizeof(int32_t));

    printf("buffer output addr: %p\n", d);
    output[0][0] = d;
}

void buffer_init(module_instance_t* instance,
                adsp_bump_allocator_t* allocator,
                uint8_t id,
                int n_inputs,
                int n_outputs,
                int frame_size)
{
    xassert(n_inputs == 1 && "buffer should have the same number of inputs and outputs");
    buffer_state_t *state = instance->state;
    buffer_constants_t* constants = instance->constants;

    memset(state, 0, sizeof(buffer_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->buffer = ADSP_BUMP_ALLOCATOR_WORD_ALLIGNED_MALLOC(allocator, n_inputs * sizeof(buffer_t));

    // point to shared memory
    state->buffer->buffer_data = constants->shared_memory;
    state->buffer->buffer_len = constants->buffer_len;
}

void buffer_control(void *state, module_control_t *control)
{
    // FIR cannot be updated, it must be set at init
}
