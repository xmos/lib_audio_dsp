// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/reverb.h"
#include "reverb_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    reverb_room_t reverb_room;
    int n_inputs;
    int n_outputs;
    int frame_size;
}reverb_state_t;

#define REVERB_REQUIRED_MEMORY(N_IN, N_OUT, FRAME_SIZE) (ADSP_RV_HEAP_SZ(MAX_SAMPLING_FREQ, MAX_ROOM_SIZE))

void reverb_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void reverb_process(int32_t **input, int32_t **output, void *app_data_state);

void reverb_control(void *state, module_control_t *control);
