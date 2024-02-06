#include <string.h>
#include <stdlib.h>
#include <xcore/assert.h>
#include <debug_print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include "stages/fork.h"
#include <stdio.h>
#include "print.h"
void fork_process(int32_t **input, int32_t **output, void *app_data_state)
{
    fork_state_t *state = app_data_state;

    int output_idx = 0;
    for(int fork_count = 0; fork_count < state->n_forks; fork_count++) {
        for(int input_index = 0; input_index < state->n_inputs; ++input_index) {
            int32_t *in = input[input_index];
            int32_t *out = output[output_idx++];
            memcpy(out, in, sizeof(int32_t) * state->frame_size);
        }
    }

}

module_instance_t* fork_init(uint8_t id, int n_inputs, int n_outputs, int frame_size, void* module_config)
{
    module_instance_t *module_instance = malloc(sizeof(module_instance_t));

    fork_state_t *state = malloc(sizeof(fork_state_t)); // malloc_from_heap
    fork_config_t *config = malloc(sizeof(fork_config_t)); // malloc_from_heap

    memset(state, 0, sizeof(fork_state_t));
    state->n_inputs = n_inputs;
    state->n_outputs = n_outputs;
    state->frame_size = frame_size;
    state->n_forks = n_outputs / n_inputs;
    xassert(n_outputs % n_inputs == 0); // must be able to fork all the inputs

    if(module_config != NULL)
    {
        fork_config_t *init_config = module_config;
        memcpy(&state->config, init_config, sizeof(fork_config_t));
    }
    else
    {
        state->config = (fork_config_t){ 0 };
    }

    memcpy(config, &state->config, sizeof(fork_config_t));

    module_instance->state = state;

    // Control stuff
    module_instance->control.config = config;
    module_instance->control.id = id;
    module_instance->control.module_type = e_dsp_stage_fork;
    module_instance->control.num_control_commands = NUM_CMDS_FORK;
    module_instance->control.config_rw_state = config_none_pending;
    return module_instance;
}

void fork_control(void *module_state, module_control_t *control)
{
    // nothing to do
}