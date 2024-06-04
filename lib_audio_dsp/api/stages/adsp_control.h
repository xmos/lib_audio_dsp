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
#include <stddef.h>
#include "adsp_module.h"
#include "adsp_pipeline.h"
#include "swlock.h"

/// The command to execute. Specifies which stage, what command and contains the buffer
/// to read from or write to.
typedef struct
{
    /// The ID of the stage to target. Consider setting the label parameter in the pipeline
    /// definition to ensure that a usable identifier gets generated for using with control.
    uint8_t instance_id;

    /// "See the generated cmds.h for the available commands. Make sure to use a command
    /// which is supported for the target stage.
    uint8_t cmd_id;

    /// Length of the command in bytes.
    uint16_t payload_len;

    /// The buffer. Must be set to a valid array of size payload_len before calling
    /// the read or write functions.
    void *payload;
}adsp_stage_control_cmd_t;

/// Control status.
typedef enum
{
    ADSP_CONTROL_SUCCESS,  ///< Command succesfully executed.
    ADSP_CONTROL_BUSY,  ///< Stage has not yet processed the command, call again.
}adsp_control_status_t;

/// Object used to control a DSP pipeline.
///
/// As there may be multiple threads attempting to interact with the DSP pipeline at
/// the same time, a separate instance of @ref adsp_controller_t must be used by each
/// to ensure that control can proceed safely.
///
/// Initialise each instance of @ref adsp_controller_t with @ref adsp_controller_init.
typedef struct {
    /// @privatesection
    module_instance_t* modules;
    size_t num_modules;
} adsp_controller_t;


/// Create a DSP controller instance for a particular pipeline.
///
/// @param ctrl The controller instance to initialise.
/// @param pipeline The DSP pipeline that will be controlled with this controller.
void adsp_controller_init(adsp_controller_t* ctrl, adsp_pipeline_t* pipeline);

/// Initiate a read command by passing in an intialised @ref adsp_stage_control_cmd_t.
///
/// Must be called repeatedly with the same cmd until ADSP_CONTROL_SUCCESS is returned. If the caller
/// abandons the attempt to read before SUCCESS is returned then this will leave the stage in a state
/// where it can never be read from again.
///
/// @param ctrl An instance of adsp_controller_t which has been initialised to control the DSP pipeline.
/// @param cmd An initialised @ref adsp_stage_control_cmd_t.
/// @return @ref adsp_control_status_t
adsp_control_status_t adsp_read_module_config(
        adsp_controller_t* ctrl,
        adsp_stage_control_cmd_t *cmd
);

/// Initiate a write command by passing in an initialised @ref adsp_stage_control_cmd_t.
///
/// Must be called repeatedly with the same cmd until ADSP_CONTROL_SUCCESS is returned.
///
/// @param ctrl An instance of adsp_controller_t which has been initialised to control the DSP pipeline.
/// @param cmd An initialised @ref adsp_stage_control_cmd_t.
/// @return @ref adsp_control_status_t
adsp_control_status_t adsp_write_module_config(
        adsp_controller_t* ctrl,
        adsp_stage_control_cmd_t *cmd
);

