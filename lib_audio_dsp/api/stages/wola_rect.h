// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "dsp/fir.h"

// Normally we would #include "wola_rect_config.h", but this module has
// no runtime configurable parameters
#include "bump_allocator.h"

// typedef struct {
//     int32_t wola_rect_len;
//     int32_t* wola_rect_data;
// } wola_rect_t;

typedef struct {
} wola_rect_config_t;


typedef struct
{
    int n_inputs;
    int n_outputs;
    int frame_size;
    int win_start;
}wola_rect_state_t;

typedef struct
{
    int32_t win_start;
}wola_rect_constants_t;




#define WOLA_RECT_STAGE_REQUIRED_MEMORY 0

void wola_rect_init(module_instance_t* instance,
                 adsp_bump_allocator_t* allocator,
                 uint8_t id,
                 int n_inputs,
                 int n_outputs,
                 int frame_size);

void wola_rect_process(int32_t ***input, int32_t **output, void *app_data_state);

void wola_rect_control(void *state, module_control_t *control);
