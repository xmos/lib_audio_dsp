// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <stdint.h>
#include "dsp/adsp.h"
#include "reverb_room_st_config.h" // Autogenerated
#include "bump_allocator.h"

typedef struct
{
    reverb_room_st_t rv;
    int n_inputs;
    int n_outputs;
    int frame_size;
}reverb_room_st_state_t;

typedef struct
{
    uint32_t sampling_freq;
    float max_room_size;
    float max_predelay;
} reverb_room_st_constants_t;

#define REVERB_ROOM_ST_STAGE_REQUIRED_MEMORY(FS, MAX_ROOM_SIZE, PD) (ADSP_BUMP_ALLOCATOR_WORD_N_BYTES(REVERB_ROOM_ST_DSP_REQUIRED_MEMORY(FS, MAX_ROOM_SIZE, PD)))

void reverb_room_st_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void reverb_room_st_process(int32_t **input, int32_t **output, void *app_data_state);

void reverb_room_st_control(void *state, module_control_t *control);
