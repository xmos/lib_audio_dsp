#pragma once

#include "adsp_module.h"
#include "dsp_thread_config.h" // Autogenerated

#define BIQUADS_MAX_CHANNELS (4)

typedef struct
{
    uint32_t max_cycles;
}dsp_thread_state_t;

module_instance_t* dsp_thread_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config);
void dsp_thread_control(void *state, module_control_t *control);
