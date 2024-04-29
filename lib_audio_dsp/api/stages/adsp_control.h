// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/// @file
///
/// The control API for the generated DSP.
///
/// These functions can be executed on any thread which is on the same tile as the
/// generated DSP threads.
///

#pragma once
#include <stdint.h>
#include "stages/adsp_module.h"

/// The command to execute. Specified which stage, what command and contains the buffer
/// to read from or write to.
typedef struct
{
    /// The ID of the stage to target. Consider setting the label parameter in the pipeline
    /// definition to ensure that a usable identifier gets generated for using with control.
    uint8_t instance_id;

    /// See cmds.h that will be generated for the available commands. Make sure to use a command
    /// which is supported for the target stage.
    uint8_t cmd_id;

    /// Length of the command in bytes.
    uint16_t payload_len;

    /// The buffer. Must be set to a valid array of size payload_len before calling
    /// the read or write functions.
    int8_t *payload;
}adsp_stage_control_cmd_t;

/// Control status.
typedef enum
{
    ADSP_CONTROL_SUCCESS,  ///< Command succesfully executed.
    ADSP_CONTROL_BUSY,  ///< Stage has not yet processed the command, call again.
    ADSP_CONTROL_ERROR  ///< An error occured.
}adsp_control_status_t;

/// Initiate a read command by passing in a filled in @ref adsp_stage_control_cmd_t.
///
/// Must be called repeatedly with the same cmd until ADSP_CONTROL_SUCCESS is returned.
///
/// @param modules A pointer to the array of modules contained within @ref adsp_pipeline_t.
/// @param num_modules Size of the num_modules array.
/// @param cmd A filled in @ref adsp_stage_control_cmd_t.
/// @return @ref adsp_control_status_t
adsp_control_status_t adsp_read_module_config(module_instance_t* modules,
                                            size_t num_modules,
                                            adsp_stage_control_cmd_t *cmd
                                        );

/// Initiate a write command by passing in a filled in @ref adsp_stage_control_cmd_t.
///
/// Must be called repeatedly with the same cmd until ADSP_CONTROL_SUCCESS is returned.
///
/// @param modules A pointer to the array of modules contained within @ref adsp_pipeline_t.
/// @param num_modules Size of the num_modules array.
/// @param cmd A filled in @ref adsp_stage_control_cmd_t.
/// @return @ref adsp_control_status_t
adsp_control_status_t adsp_write_module_config(module_instance_t* modules,
                                            size_t num_modules,
                                            adsp_stage_control_cmd_t *cmd
                                        );

