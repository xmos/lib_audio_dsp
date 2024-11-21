// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/bfp_multiply.h"

void bfp_multiply_process(complex_spectrum_t **input, complex_spectrum_t **output, void *app_data_state)
{
    bfp_multiply_state_t *state = app_data_state;

    bfp_complex_s32_t out_spect;
    if(input[0] == output[0]){
        bfp_complex_s32_init(&out_spect, input[0]->data, input[0]->exp, (state->nfft >> 1) + 1, 0);
    }
    else if (input[1] == output[0]){
        bfp_complex_s32_init(&out_spect, input[1]->data, input[1]->exp, (state->nfft >> 1) + 1, 0);
    }
    else {
        bfp_complex_s32_init(&out_spect, output[0]->data, state->exp, (state->nfft >> 1) + 1, 0);
    }

    output[0][0] = input[0][0];

    bfp_complex_s32_t in_spect_0 , in_spect_1;
    bfp_complex_s32_init(&in_spect_0, input[0]->data, input[0]->exp, state->nfft >> 1, 1);
    bfp_complex_s32_init(&in_spect_1, input[1]->data, input[1]->exp, state->nfft >> 1, 1);

    bfp_complex_s32_mul(&out_spect, &in_spect_0, &in_spect_1);

    output[0]->exp = out_spect.exp;

}

void bfp_multiply_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    bfp_multiply_state_t *state = instance->state;
    bfp_multiply_constants_t* constants = instance->constants;

    memset(state, 0, sizeof(bfp_multiply_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->nfft = constants->nfft;
    state->exp = constants->exp;

    xassert(n_outputs == 1 && "Multiply should only have one output");
}


