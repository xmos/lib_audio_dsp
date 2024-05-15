// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "app_dsp.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "stdbool.h"
#include <xcore/assert.h>
#include <assert.h>
#include "xcore/chanend.h"
#include "xcore/parallel.h"

//#include "dspt_control.h"
#include "print.h"
#include <stages/adsp_pipeline.h>
#include <adsp_generated_auto.h>

#include "run_cmds.h"

static adsp_pipeline_t * m_dsp;

// send data to dsp
void app_dsp_source(int32_t** data) {
    adsp_pipeline_source(m_dsp, data);
}

// read output
void app_dsp_sink(int32_t** data) {
    adsp_pipeline_sink(m_dsp, data);
}

DECLARE_JOB(adsp_auto_pipeline_main, (adsp_pipeline_t*));
DECLARE_JOB(dsp_control_thread, (chanend_t, module_instance_t*, size_t));

void dsp_control_thread(chanend_t c_control, module_instance_t* modules, size_t num_modules)
{
    chan_in_word(c_control);
    send_control_cmds(m_dsp, c_control);
}

// do dsp
void app_dsp_main(chanend_t c_control) {
    m_dsp = adsp_auto_pipeline_init();

    PAR_JOBS(
        PJOB(adsp_auto_pipeline_main, (m_dsp)),
        PJOB(dsp_control_thread, (c_control, m_dsp->modules, m_dsp->n_modules))
    );
}

int app_dsp_frame_size() {
    while(!m_dsp);
    return m_dsp->input_mux.chan_cfg[0].frame_size;
}
