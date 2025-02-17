// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/pipeline.h"

void pipeline_init(module_instance_t* instance, adsp_bump_allocator_t* allocator, uint8_t id, int n_inputs, int n_outputs, int frame_size)
{
    pipeline_state_t *state = instance->state;
    pipeline_config_t *config = instance->control.config;

    memcpy(state->checksum, config->checksum, sizeof(state->checksum));
}

void pipeline_control(void *module_state, module_control_t *control)
{
    if(control->config_rw_state == config_write_pending)
    {
        // No write commands for the pipeline stage
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        // The only supported command is checksum and that doesn't change after initialisation so do nothing
        control->config_rw_state = config_read_updated;
    }
    else
    {
        // nothing to do.
    }
}
