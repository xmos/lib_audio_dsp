// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/ifft.h"

void ifft_process(int32_t ** input, int32_t ** output, void * app_data_state)
{
    xassert(app_data_state != NULL);
    ifft_state_t *state = app_data_state;
    bfp_s32_t *time_domain_result = bfp_fft_inverse_mono(input[0]);

    //denormalise and escape BFP domain
    bfp_s32_use_exponent(time_domain_result, state->ifft[0]->exp);
    output[0] = time_domain_result->data;
}

void ifft_init(module_instance_t* instance,
                adsp_bump_allocator_t* allocator,
                uint8_t id,
                int n_inputs,
                int n_outputs,
                int frame_size)
{
    xassert(n_inputs == 1 && "ifft should have the same number of inputs and outputs");
    ifft_state_t *state = instance->state;
    ifft_constants_t* constants = instance->constants;

    memset(state, 0, sizeof(ifft_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    state->ifft = ADSP_BUMP_ALLOCATOR_WORD_ALLIGNED_MALLOC(allocator, n_inputs * sizeof(ifft_t));

    // point to shared memory
    state->ifft[0]->data = constants->shared_memory;
    state->ifft[0]->nfft = constants->nfft;
    state->ifft[0]->exp = constants->exp;

}

void ifft_control(void *state, module_control_t *control)
{
    // FIR cannot be updated, it must be set at init
}
