// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "adsp_pipeline.h"

// Function to write and read back control parameters.
// Only the commands included in the stage config are used.
void send_control_cmds(adsp_pipeline_t * m_dsp, chanend_t c_control);
