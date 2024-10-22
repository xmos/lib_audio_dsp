// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <stdint.h>
#include "dsp/adsp.h"
#include "reverb_plate_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    reverb_plate_t rv;
    int n_inputs;
    int n_outputs;
    int frame_size;
}reverb_plate_state_t;

typedef struct
{
    uint32_t sampling_freq;
    float max_predelay;
} reverb_plate_constants_t;

#define REVERB_PLATE_STAGE_REQUIRED_MEMORY(FS, PD) (ADSP_BUMP_ALLOCATOR_WORD_N_BYTES(REVERB_PLATE_DSP_REQUIRED_MEMORY(FS, PD)))

void reverb_plate_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void reverb_plate_process(int32_t **input, int32_t **output, void *app_data_state);

void reverb_plate_control(void *state, module_control_t *control);
