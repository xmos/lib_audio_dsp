#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/cascaded_biquads.h"

DSP_MODULE_PROCESS_ATTR
void cascaded_biquads_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    cascaded_biquads_state_t *state = app_data_state;

    // do while saves instructions for cases
    // where the loop will always execute at
    // least once
    int i = 0;
    do {
        int32_t *in = input[i];
        int32_t *out = output[i];
        
        int j = 0;
        do {
            *out++ = adsp_cascaded_biquads_8b((*in++),
                        state->config.filter_coeffs,
                        state->filter_states[i],
                        state->config.left_shift);
        } while(++j < state->frame_size);
    } while(++i < state->n_outputs);
}

DSP_MODULE_INIT_ATTR
module_instance_t* cascaded_biquads_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config)
{
    module_instance_t *module_instance = malloc(sizeof(module_instance_t));

    cascaded_biquads_state_t *state = malloc(sizeof(cascaded_biquads_state_t)); // malloc_from_heap
    cascaded_biquads_config_t *config = malloc(sizeof(cascaded_biquads_config_t)); // malloc_from_heap

    memset(state, 0, sizeof(cascaded_biquads_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    if(module_config != NULL)
    {
        cascaded_biquads_config_t *init_config = module_config;
        memcpy(&state->config, init_config, sizeof(cascaded_biquads_config_t));
    }
    else
    {
        // b2 / a0 b1 / a0 b0 / a0 -a1 / a0 -a2 / a0
        int32_t DWORD_ALIGNED filter_coeffs [40] = {0};
        filter_coeffs[0] = 1073741824;
        filter_coeffs[5] = 1073741824;
        filter_coeffs[10] = 1073741824;
        filter_coeffs[15] = 1073741824;
        filter_coeffs[20] = 1073741824;
        filter_coeffs[25] = 1073741824;
        filter_coeffs[30] = 1073741824;
        filter_coeffs[35] = 1073741824;

        memcpy(state->config.filter_coeffs, filter_coeffs, sizeof(filter_coeffs));
    }

    memcpy(config, &state->config, sizeof(cascaded_biquads_config_t));

    module_instance->state = state;
    module_instance->process_sample = cascaded_biquads_process;

    // Control stuff
    module_instance->module_control = cascaded_biquads_control;
    module_instance->control.config = config;
    module_instance->control.id = id;
    module_instance->control.module_type = e_dsp_stage_cascaded_biquads;
    module_instance->control.num_control_commands = NUM_CMDS_BIQUAD;
    module_instance->control.config_rw_state = config_none_pending;
    return module_instance;
}

DSP_MODULE_CONTROL_ATTR
void cascaded_biquads_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    cascaded_biquads_state_t *state = module_state;
    xassert(control != NULL);
    cascaded_biquads_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(cascaded_biquads_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(cascaded_biquads_config_t));
        control->config_rw_state = config_read_updated;
    }
}