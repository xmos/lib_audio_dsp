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

DECLARE_JOB(fileio_task, (chanend_t));
DECLARE_JOB(app_dsp_main, (chanend_t));

int main_c()
{
    channel_t c_control = chan_alloc();

    PAR_JOBS(
        PJOB(fileio_task, (c_control.end_a)),
        PJOB(app_dsp_main, (c_control.end_b))
    );

    return 0;
}
