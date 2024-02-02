

#pragma once
#include <xccompat.h>
#include <stdint.h>

#define MAX_CHANNELS (4)
// XC safe wrapper for adsp

void app_dsp_init(void);

// send data to dsp
void app_dsp_source(REFERENCE_PARAM(int32_t, data), int num_channels);

// read output
void app_dsp_sink(REFERENCE_PARAM(int32_t, data), int num_channels);

// do dsp
void app_dsp_main(chanend c_control);


// TODO control


