#ifndef PARAMETRIC_EQ_H
#define PARAMETRIC_EQ_H

#include "adsp_module.h"
#include "parametric_eq_config.h" // Autogenerated

#define MAX_CHANNELS (4)
#define DSP_NUM_STATES_PER_BIQUAD 4
// Go in DSP module file

typedef struct
{
    parametric_eq_config_t config;
    int32_t DWORD_ALIGNED filter_states[MAX_CHANNELS][FILTERS * DSP_NUM_STATES_PER_BIQUAD];
    int n_inputs;
    int n_outputs;
    int frame_size;
}parametric_eq_state_t;

// Public functions
DSP_MODULE_INIT_ATTR module_instance_t* parametric_eq_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config);

DSP_MODULE_PROCESS_ATTR  void parametric_eq_process(int32_t *input, int32_t *output, void *app_data_state);

DSP_MODULE_CONTROL_ATTR void parametric_eq_control(void *state, module_control_t *control);

#endif
