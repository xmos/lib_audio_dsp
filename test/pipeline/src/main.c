// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <xcore/assert.h>
#include <xcore/parallel.h>
#include <xcore/channel.h>
#include <xcore/chanend.h>
//#include "adsp_pipeline.h"
#include <adsp_generated_auto.h>

DECLARE_JOB(fileio_task, (chanend_t));
DECLARE_JOB(app_dsp_main, (chanend_t));
//DECLARE_JOB(app_ctrl, (adsp_pipeline_t*, chanend_t));

//adsp_pipeline_t * m_dsp;

int main_c()
{
    channel_t c_control = chan_alloc();
    //m_dsp = adsp_auto_pipeline_init();

    PAR_JOBS(

        PJOB(fileio_task, (c_control.end_a)),
        PJOB(app_dsp_main, (c_control.end_b))
        //PJOB(app_ctrl, (m_dsp, c_control.end_b))
    );

    return 0;
}
