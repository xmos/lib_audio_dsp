
#include "app_dsp.h"
#include "stdbool.h"
#include "xcore/chanend.h"
#include "xcore/parallel.h"
//#include "dspt_control.h"

#include <stages/adsp_pipeline.h>

static audio_dsp_t m_dsp;

// send data to dsp
void app_dsp_source(REFERENCE_PARAM(int32_t, data), int num_channels) {
    int32_t* in_data[MAX_CHANNELS] = {0};
    for(int i=0; i<num_channels; i++)
    {
        in_data[i] = &data[i];
    };

    adsp_pipeline_source(&m_dsp, in_data);
}

// read output
void app_dsp_sink(REFERENCE_PARAM(int32_t, data), int num_channels) {
    int32_t* out_data[MAX_CHANNELS] = {0};
    for(int i=0; i<num_channels; i++)
    {
        out_data[i] = &data[i];
    }
    adsp_pipeline_sink(&m_dsp, out_data);
}



// do dsp
void app_dsp_main(chanend_t c_control) {
    adsp_pipeline_init(&m_dsp);

    adsp_module_array_t modules = adsp_pipeline_get_modules(&m_dsp);
    (void)modules;
    PAR_JOBS(
        PJOB(adsp_pipeline_main, (&m_dsp))
        //PJOB(dsp_control_thread, (c_control, modules.modules, modules.num_modules)) // TODO
    );
    adsp_pipeline_main(&m_dsp);
}

