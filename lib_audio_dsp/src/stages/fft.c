// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/fft.h"

void fft_process(int32_t ** input, bfp_complex_s32_t *** output, void * app_data_state)
{
    xassert(app_data_state != NULL);
    fft_state_t *state = app_data_state;
    // put signal int32_t array into bfp

    printf("input[0] addr: %p\n", &input[0][0]);
    state->fft[0].data = &input[0][0];
    printf("fft buffer addr: %p\n", state->fft[0].data);

    bfp_s32_init(&(state->fft[0].signal), &input[0][0], state->fft[0].exp, state->fft[0].nfft, 1);

    printf("doing fft\n");
    // do the FFT
    bfp_complex_s32_t * c = bfp_fft_forward_mono(&(state->fft[0].signal));

    printf("fft output data addr: %p\n", c->data);

    output[0][0] = c;
}

void fft_init(module_instance_t* instance,
                adsp_bump_allocator_t* allocator,
                uint8_t id,
                int n_inputs,
                int n_outputs,
                int frame_size)
{
    xassert(n_inputs == 1 && "fft should have the same number of inputs and outputs");
    fft_state_t *state = instance->state;
    fft_constants_t* constants = instance->constants;

    memset(state, 0, sizeof(fft_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->fft = ADSP_BUMP_ALLOCATOR_WORD_ALLIGNED_MALLOC(allocator, n_inputs * sizeof(fft_t));

    // point to shared memory
    state->fft[0].nfft = constants->nfft;
    state->fft[0].exp = constants->exp;

}

void fft_control(void *state, module_control_t *control)
{
    // FIR cannot be updated, it must be set at init
}
