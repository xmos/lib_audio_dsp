// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "bump_allocator.h"
#include "dsp/drc.h"
#include "envelope_detector_peak_config.h" // Autogenerated

typedef struct
{
    env_detector_t *env_det;
    int n_inputs;
    int n_outputs;
    int frame_size;
}envelope_detector_peak_state_t;

#define ENVELOPE_DETECTOR_PEAK_STAGE_REQUIRED_MEMORY(N_IN) (N_IN * sizeof(env_detector_t))

void envelope_detector_peak_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size);

void envelope_detector_peak_process(int32_t **input, int32_t **output, void *app_data_state);

void envelope_detector_peak_control(void *state, module_control_t *control);
