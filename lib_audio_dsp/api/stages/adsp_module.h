// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <xmath/xmath.h>

#define DWORD_ALIGNED_MALLOC(n_bytes) (void*)(((((uint64_t)malloc((n_bytes) + 7)) + 7) >> 3) << 3) // This is allocating more memory than requested to sort out dword alignment.
                                                                                                   // This is done assuming that this memory is never freed.

typedef enum
{
    config_read_pending,    // Control waiting to read the updated config from DSP
    config_write_pending,   // Config written by control and waiting for DSP to update
    config_read_updated,    // Config updated with the latest in order to service a read command
    config_none_pending     // All done. Control and DSP not waiting on anything.
}config_rw_state_t;

// Control related information shared between control thread and DSP
typedef struct
{
    void *config;
    uint32_t id;    // Unique module identifier assigned by the host
    uint32_t num_control_commands;
    uint8_t module_type;
    uint8_t cmd_id;
    config_rw_state_t config_rw_state;
}module_control_t;

typedef struct
{
    void *state;    // Pointer to the module's state memory
    module_control_t control;
}module_instance_t;



typedef struct
{
    uint8_t instance_id;
}module_info_t;

