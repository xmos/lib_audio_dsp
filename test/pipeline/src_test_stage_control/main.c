// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <xcore/assert.h>
#include <assert.h>
#include <xcore/parallel.h>
#include <xcore/channel.h>
#include <xcore/chanend.h>

#include <stages/adsp_pipeline.h>
#include <stages/adsp_control.h>
#include <adsp_generated_auto.h>
#include "adsp_instance_id_auto.h"
#include "cascaded_biquads_config.h"
#include "cmds.h"


static adsp_pipeline_t * m_dsp;

DECLARE_JOB(app_control, (module_instance_t*, size_t));

#define member_size(type, member) (sizeof( ((type *)0)->member ))
void app_control(module_instance_t* modules, size_t num_modules)
{
    adsp_stage_control_cmd_t cmd;
    int8_t payload_buf[256];
    cmd.instance_id = casc_biquad_stage_index;
    cmd.cmd_id = CMD_CASCADED_BIQUADS_LEFT_SHIFT;
    cmd.payload_len = member_size(cascaded_biquads_config_t, left_shift);
    cmd.payload = payload_buf;

    // Write control command to the stage
    int left_shift_write[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(cmd.payload, left_shift_write, cmd.payload_len);

    adsp_control_status_t ret = ADSP_CONTROL_BUSY;
    do {
        ret = adsp_write_module_config(modules, num_modules, &cmd);
    }while(ret == ADSP_CONTROL_BUSY);

    assert(ret == ADSP_CONTROL_SUCCESS);

    memset(cmd.payload, 0, sizeof(payload_buf));

    // Read back the written data
    ret = ADSP_CONTROL_BUSY;
    do {
        ret = adsp_read_module_config(modules, num_modules, &cmd);
    }while(ret == ADSP_CONTROL_BUSY);

    assert(ret == ADSP_CONTROL_SUCCESS);

    int left_shift_read[8];
    memcpy(left_shift_read, cmd.payload, cmd.payload_len);

    for(int i=0; i<8; i++)
    {
        if(left_shift_read[i] != left_shift_write[i])
        {
            printf("Mismatch at index %d. Expected %d, found %d\n", i, left_shift_write[i], left_shift_read[i]);
            assert(0);
        }
    }
    _Exit(0);
}

int main(int argc, char **argv) {
    m_dsp = adsp_auto_pipeline_init();

    PAR_JOBS(
        PJOB(adsp_auto_pipeline_main, (m_dsp)),
        PJOB(app_control, (m_dsp->modules, m_dsp->n_modules))
    );
}
