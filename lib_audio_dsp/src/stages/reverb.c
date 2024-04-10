// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/reverb.h"

void reverb_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size)
{
    xassert(n_inputs == n_outputs && "Reverb should have the same number of inputs and outputs");
    reverb_state_t *state = instance->state;
    reverb_config_t *config = instance->control.config;

    memset(state, 0, sizeof(reverb_state_t));

    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    float fs = config->sampling_freq;
    float max_room_size = config->max_room_size;

    float const room_size = config->room_size;
    float const decay = config->decay;
    float const damping = config->damping;
    int32_t wet_gain = config->wet_gain;
    int32_t dry_gain = config->dry_gain;
    float const pregain = config->pregain;

    // Both fs and max_room_size are used in heap memory calculation, which is currently defined at compile time
    // #define REVERB_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (RV_HEAP_SZ(48000, 1.0f)), so ensure the fs and max_room_size
    // we get at initialisation match.

    xassert(fs <= (float)ADSP_RV_MAX_SAMPLING_FREQ);
    xassert(max_room_size <= (float)ADSP_RV_MAX_ROOM_SIZE);

    xassert(n_inputs == 1); // Currently support only 1 channel reverb

    uint32_t sz = ADSP_RV_HEAP_SZ(fs, max_room_size);
    uint8_t *reverb_heap = adsp_bump_allocator_malloc(allocator, sz);
    memset(reverb_heap, 0, sz);

    state->reverb_room = adsp_reverb_room_init(fs,
                                max_room_size, room_size,
                                decay, damping, wet_gain,
                                dry_gain, pregain,
                                reverb_heap);
}

void reverb_process(int32_t **input, int32_t **output, void *app_data_state)
{
    reverb_state_t *state = app_data_state;
    int32_t *in = input[0];
    int32_t *out = output[0];
    int j = 0;
    do
    {
        *out++ = adsp_reverb_room(&state->reverb_room, (*in++));
    } while (++j < state->frame_size);
}

void reverb_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    reverb_state_t *state = module_state;
    xassert(control != NULL);
    reverb_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        state->reverb_room.wet_gain = config->wet_gain; // Only setting the wet gain supported for now
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        config->wet_gain = state->reverb_room.wet_gain; // wet_gain, being the only writable parameter is expected to change
        control->config_rw_state = config_read_updated;
    }
    else {
        // nothing to do
    }

}
