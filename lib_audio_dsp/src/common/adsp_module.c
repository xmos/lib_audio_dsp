#include <xcore/channel.h>
#include "adsp_module.h"

static void do_control(module_instance_t** modules, size_t num_modules)
{
    for(size_t i=0; i<num_modules; i++)
    {
        modules[i]->module_control(modules[i]->state, &modules[i]->control);
    }
}

#pragma stackfunction 1000 // TODO
void dsp_thread(chanend_t c_source, chanend_t c_dest, module_instance_t** modules, size_t num_modules)
{
    int32_t input_data[DSP_INPUT_CHANNELS] = {0};
    int32_t output_data[DSP_OUTPUT_CHANNELS] = {0};

    while(1)
    {
        int32_t *input_ptr = input_data;
        int32_t *output_ptr = output_data;
        chan_in_buf_word(c_source, (uint32_t*)input_ptr, DSP_INPUT_CHANNELS);
        for(int i=0; i<num_modules; i++)
        {
            modules[i]->process_sample(input_ptr, output_ptr, modules[i]->state);

            if(i < num_modules-1) // If we have more iterations
            {
                int32_t *temp = input_ptr;
                input_ptr = output_ptr;
                output_ptr = temp;
            }
        }
        chan_out_buf_word(c_dest, (uint32_t*)output_ptr, DSP_OUTPUT_CHANNELS);

        do_control(modules, num_modules);
    }
}

