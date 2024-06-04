// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

// Minimal stage required to process some control.

#pragma once

#include <stages/bump_allocator.h>
#include <stages/adsp_module.h>
#include <string.h>
#include <dummy_config.h>

typedef struct {
    dummy_config_t config;
} dummy_state_t;

#define DUMMY_STAGE_REQUIRED_MEMORY 0

static inline void dummy_init(module_instance_t* state, adsp_bump_allocator_t* alloc, int index, int nin, int nout, int chan) {

}

static inline void dummy_process(int32_t **input, int32_t **output, void *app_data_state) {}

static inline void dummy_control(dummy_state_t *module_state, module_control_t *control) {

    dummy_state_t *state = module_state;
    dummy_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(dummy_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(dummy_config_t));
        control->config_rw_state = config_read_updated;
    }
}
