// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/// Custom stage which fills the output with the frame index of that sample
/// this is used in test_frame_size.py to check the generated pipeline
/// really has the correct frame size.

#include <stages/bump_allocator.h>

typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
}frame_count_state_t;

#define FRAME_COUNT_STAGE_REQUIRED_MEMORY 0

void frame_count_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size) {
    frame_count_state_t *state = instance->state;

    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
}

void frame_count_process(int32_t **input, int32_t **output, void *app_data_state) {
    frame_count_state_t *state = app_data_state;
    for(int i = 0; i < state->n_inputs; ++i) {
        for(int j = 0; j < state->frame_size; ++j) {
            output[i][j] = j;
        }
    }
}
