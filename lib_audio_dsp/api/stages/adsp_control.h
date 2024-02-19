// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once
#include <stdint.h>
#include "stages/adsp_module.h"

// All the Control command information
typedef struct
{
    uint8_t instance_id;
    uint8_t cmd_id;
    uint16_t payload_len;
    int8_t *payload;
}adsp_stage_control_cmd_t;

typedef enum
{
    ADSP_CONTROL_SUCCESS,
    ADSP_CONTROL_BUSY,
    ADSP_CONTROL_ERROR
}adsp_control_status_t;

// Read a module instance's config structure for a given command ID
adsp_control_status_t adsp_read_module_config(module_instance_t* modules, // Array of module instance pointers
                                            size_t num_modules, // Total number of modules
                                            adsp_stage_control_cmd_t *cmd
                                        );

// Write to a module instance's config structure for a given command ID
adsp_control_status_t adsp_write_module_config(module_instance_t* modules, // Array of module instance pointers
                                            size_t num_modules, // Total number of modules
                                            adsp_stage_control_cmd_t *cmd
                                        );

