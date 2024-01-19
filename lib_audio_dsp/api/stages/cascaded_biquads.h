#pragma once

#include "dsp/cascaded_biquads.h"
#include "cascaded_biquads_config.h" // Autogenerated

#define MAX_CHANNELS (4)

typedef struct
{
    cascaded_biquads_config_t config;
    int32_t DWORD_ALIGNED filter_states[MAX_CHANNELS][64];
    int n_inputs;
    int n_outputs;
    int frame_size;
}cascaded_biquads_state_t;

DSP_MODULE_INIT_ATTR module_instance_t* cascaded_biquads_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config);

DSP_MODULE_PROCESS_ATTR  void cascaded_biquads_process(int32_t **input, int32_t **output, void *app_data_state);

DSP_MODULE_CONTROL_ATTR void cascaded_biquads_control(void *state, module_control_t *control);
