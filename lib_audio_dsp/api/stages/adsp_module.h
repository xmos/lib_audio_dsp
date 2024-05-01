// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/// @file
///
/// Defines the generic structs that will hold the state and control configuration
/// for each stage.

#pragma once

#include <stdint.h>

/// Control states, used to communicate between DSP and control threads
/// to notify when control needs processing.
typedef enum
{
    config_read_pending,    ///< Control waiting to read the updated config from DSP.
    config_write_pending,   ///< Config written by control and waiting for DSP to update.
    config_read_updated,    ///< Stage has succesfully consumed a read command.
    config_none_pending     ///< All done. Control and DSP not waiting on anything.
} config_rw_state_t;

/// Control related information shared between control thread and DSP.
typedef struct
{
    void *config;  ///< Pointer to a stage-specific config struct which is used by the control thread.
    uint32_t id;  ///< Unique module identifier assigned by the host
    uint32_t num_control_commands;  ///< The number of control commands for this stage.
    uint8_t module_type;  ///< Identifies the stage type. Each type of stage has a unique identifier.
    uint8_t cmd_id;  ///< Is set to the current command being processed.
    config_rw_state_t config_rw_state;
}module_control_t;


/// The entire state of a stage in the pipeline.
typedef struct
{
    void *state;    ///< Pointer to the module's state memory.
    module_control_t control;  ///< Module's control state.
}module_instance_t;

