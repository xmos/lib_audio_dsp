// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.


#pragma once
#include <xccompat.h>
#include <stdint.h>

#define MAX_CHANNELS (4)
// XC safe wrapper for adsp

void app_dsp_init(void);

// send data to dsp
void app_dsp_source(int32_t** data);

// read output
void app_dsp_sink(int32_t** data);

// do dsp
void app_dsp_main(chanend c_control);

// get frame size
int app_dsp_frame_size();

// TODO control


