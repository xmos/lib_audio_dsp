#include <xcore/channel.h>
#include <xcore/chanend.h>
#include "stages/adsp_module.h"

static void do_control(module_instance_t** modules, size_t num_modules)
{
    for(size_t i=0; i<num_modules; i++)
    {
        modules[i]->module_control(modules[i]->state, &modules[i]->control);
    }
}

void dsp_thread(chanend_t c_source, chanend_t c_dest, module_instance_t** modules, size_t num_modules)
{
    int32_t input_data[DSP_INPUT_CHANNELS][1] = {{0}};
    int32_t output_data[DSP_OUTPUT_CHANNELS][1] = {{0}};

    int32_t *input_data_ptrs[4] = {input_data[0], input_data[1], input_data[2], input_data[3]};
    int32_t *output_data_ptrs[4] = {output_data[0], output_data[1], input_data[2], output_data[3]};
    while(1)
    {
        int32_t **input_ptr = input_data_ptrs;
        int32_t **output_ptr = output_data_ptrs;

        for(int i = 0; i < DSP_INPUT_CHANNELS; i++) {
            int x = chanend_in_word(c_source);
            input_data[i][0] = x;
        }
        chanend_check_end_token(c_source);

        for(int i=0; i<num_modules; i++)
        {
            modules[i]->process_sample(input_ptr, output_ptr, modules[i]->state);

            if(i < num_modules-1) // If we have more iterations
            {
                int32_t **temp = input_ptr;
                input_ptr = output_ptr;
                output_ptr = temp;
            }
        }

        for(int i = 0; i < DSP_OUTPUT_CHANNELS; i++) {
            chanend_out_word(c_dest, output_data[i][0]);
        }
        chanend_out_end_token(c_dest);

        do_control(modules, num_modules);
    }
}

