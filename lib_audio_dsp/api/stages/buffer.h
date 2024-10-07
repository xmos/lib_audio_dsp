// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/fir.h"

// Normally we would #include "buffer_config.h", but this module has
// no runtime configurable parameters
#include "bump_allocator.h"

typedef struct {
} buffer_config_t;

typedef struct {
    // int32_t buffer_len;
    int32_t* overlap_data;
} buffer_t;

typedef struct
{
    buffer_t *buffer;
    int n_inputs;
    int n_outputs;
    int frame_size;
    int buffer_len;
    int overlap_len;
}buffer_state_t;

typedef struct
{
    int32_t buffer_len;
}buffer_constants_t;



#define BUFFER_STAGE_REQUIRED_MEMORY(N_CH, BUFF_LEN) \
     + ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(N_CH * sizeof(buffer_t)) \
     + ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(N_CH * (BUFF_LEN) * sizeof(int32_t))

void buffer_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void buffer_process(int32_t **input, int32_t **output, void *app_data_state);

void buffer_control(void *state, module_control_t *control);
