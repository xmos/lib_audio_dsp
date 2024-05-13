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

#include "adsp_control.h"
#include "adsp_instance_id_auto.h"
#include "cmds.h"
#include "cmd_offsets.h"

//#include "dspt_control.h"
#include "print.h"
#include <stages/adsp_pipeline.h>
#include <adsp_generated_auto.h>


static adsp_pipeline_t * m_dsp;

// send data to dsp
void app_dsp_source(int32_t** data) {
    adsp_pipeline_source(m_dsp, data);
}

// read output
void app_dsp_sink(int32_t** data) {
    adsp_pipeline_sink(m_dsp, data);
}

#include "print.h"
static void send_control_cmd(adsp_pipeline_t * m_dsp, chanend_t c_control) {
    adsp_stage_control_cmd_t cmd;
    int8_t payload_buf[256];
    cmd.instance_id = thread0_stage_index;
    cmd.cmd_id = limiter_rms_config_offsets[4].cmd_id;
    cmd.payload_len = limiter_rms_config_offsets[4].size;;
    cmd.payload = payload_buf;
    //int NUM_VALUES_LIMITER_RMS_GAIN = cmd.payload_len / 4;
    #define NUM_VALUES_LIMITER_RMS_GAIN 1
    // Write control command to the stage
    int values_write[NUM_VALUES_LIMITER_RMS_GAIN] = {1};
    memcpy(cmd.payload, values_write, cmd.payload_len);
    adsp_control_status_t ret = ADSP_CONTROL_BUSY;
    printintln(111);

    do {
        ret = adsp_write_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
        printintln(ret);
    }while(ret == ADSP_CONTROL_BUSY);
    return;

    assert(ret == ADSP_CONTROL_SUCCESS);

    memset(cmd.payload, 0, sizeof(payload_buf));

    // Read back the written data
    ret = ADSP_CONTROL_BUSY;
    do {
        ret = adsp_read_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
    }while(ret == ADSP_CONTROL_BUSY);

    assert(ret == ADSP_CONTROL_SUCCESS);

    int values_read[NUM_VALUES_LIMITER_RMS_GAIN];
    memcpy(values_read, cmd.payload, cmd.payload_len);

    for(int i=0; i<NUM_VALUES_LIMITER_RMS_GAIN; i++)
    {
        if(values_read[i] != values_write[i])
        {
            printf("Command %d: mismatch at index %d. Expected %d, found %d\n", CMD_LIMITER_RMS_GAIN, i, values_write[i], values_read[i]);
            assert(0);
        }
    }
}
#include <xcore/hwtimer.h>

DECLARE_JOB(adsp_auto_pipeline_main, (adsp_pipeline_t*));
DECLARE_JOB(dsp_control_thread, (chanend_t, module_instance_t*, size_t));

void dsp_control_thread(chanend_t c_control, module_instance_t* modules, size_t num_modules)
{
    //if (0)

    /*hwtimer_t tmr = hwtimer_alloc();
    {
        xassert(tmr != 0);
        hwtimer_delay(tmr, 10000000); // 1ms with 100 MHz timer tick
    }
    hwtimer_free(tmr);*/
    chan_in_word(c_control);
    printintln(555);
    send_control_cmd(m_dsp, c_control);
}

// do dsp
void app_dsp_main(chanend_t c_control) {
    m_dsp = adsp_auto_pipeline_init();

    PAR_JOBS(
        PJOB(adsp_auto_pipeline_main, (m_dsp)),
        PJOB(dsp_control_thread, (c_control, m_dsp->modules, m_dsp->n_modules)) // TODO
    );
}

int app_dsp_frame_size() {
    while(!m_dsp);
    return m_dsp->input_mux.chan_cfg[0].frame_size;
}
