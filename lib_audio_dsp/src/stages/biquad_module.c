#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "stages/biquad_module.h"

DSP_MODULE_PROCESS_ATTR
void biquad_process(int32_t **input, int32_t **output, void *app_data_state)
{
    xassert(app_data_state != NULL);
    biquad_state_t *state = app_data_state;

    for(int i=0; i<state->n_outputs; i++)
    {
        int32_t *in = input[i];
        int32_t *out = output[i];
        for(int j=0; j<state->frame_size; j++)
        {
            *out++ = adsp_biquad((*in++),
                        state->config.filter_coeffs,
                        state->filter_states[i],
                        state->config.left_shift);
        }
    }
}

DSP_MODULE_INIT_ATTR
module_instance_t* biquad_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config)
{
    module_instance_t *module_instance = malloc(sizeof(module_instance_t));

    biquad_state_t *state = malloc(sizeof(biquad_state_t)); // malloc_from_heap
    biquad_config_t *config = malloc(sizeof(biquad_config_t)); // malloc_from_heap

    memset(state, 0, sizeof(biquad_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;

    if(module_config != NULL)
    {
        biquad_config_t *init_config = module_config;
        memcpy(&state->config, init_config, sizeof(biquad_config_t));
    }
    else
    {
        // b2 / a0 b1 / a0 b0 / a0 -a1 / a0 -a2 / a0
        const int32_t DWORD_ALIGNED filter_coeffs [5] = {
            1073741824 , 0 , 0 , 0 , 0 ,
        };
        memcpy(state->config.filter_coeffs, filter_coeffs, sizeof(filter_coeffs));
    }

    memcpy(config, &state->config, sizeof(biquad_config_t));

    module_instance->state = state;
    module_instance->process_sample = biquad_process;

    // Control stuff
    module_instance->module_control = biquad_control;
    module_instance->control.config = config;
    module_instance->control.id = id;
    module_instance->control.module_type = biquad;
    module_instance->control.num_control_commands = NUM_CMDS_BIQUAD;
    module_instance->control.config_rw_state = config_none_pending;
    return module_instance;
}

DSP_MODULE_CONTROL_ATTR
void biquad_control(void *module_state, module_control_t *control)
{
    xassert(module_state != NULL);
    biquad_state_t *state = module_state;
    xassert(control != NULL);
    biquad_config_t *config = control->config;

    if(control->config_rw_state == config_write_pending)
    {
        // Finish the write by updating the working copy with the new config
        memcpy(&state->config, config, sizeof(biquad_config_t));
        control->config_rw_state = config_none_pending;
    }
    else if(control->config_rw_state == config_read_pending)
    {
        memcpy(config, &state->config, sizeof(biquad_config_t));
        control->config_rw_state = config_read_updated;
    }
}
