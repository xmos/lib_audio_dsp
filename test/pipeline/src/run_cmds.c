// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if SEND_TEST_CONTROL_COMMANDS

#include <string.h>
#include "adsp_control.h"
#include "adsp_instance_id_auto.h"
#include "xcore/hwtimer.h"
#include <xcore/assert.h>
#include <debug_print.h>

#include "run_cmds.h"

#include "control_test_params.h"

#define CONTROL_COMMAND_TIMEOUT_TICKS 100000 // one tick is 10ns
void send_control_cmds(adsp_pipeline_t * m_dsp, chanend_t c_control) {

    adsp_stage_control_cmd_t cmd;
    int8_t payload_buf[CMD_PAYLOAD_MAX_SIZE];
    cmd.instance_id = control_stage_index;
    hwtimer_t t = hwtimer_alloc();
    uint8_t values_write[CMD_PAYLOAD_MAX_SIZE];
    uint8_t values_read[CMD_PAYLOAD_MAX_SIZE];

    for (int cmd_idx = 0; cmd_idx<CMD_TOTAL_NUM; cmd_idx++)
    {

        // Fill up the command fields
        cmd.cmd_id = control_config[cmd_idx].cmd_id;
        cmd.payload_len = control_config[cmd_idx].cmd_size;
        cmd.payload = payload_buf;
        memset(cmd.payload, 0, cmd.payload_len);

        // Write control command to the stage
        memcpy(cmd.payload, control_config[cmd_idx].payload, cmd.payload_len);

        // Save the payload values for the final check
        memcpy(values_write, cmd.payload, cmd.payload_len);

        adsp_control_status_t ret = ADSP_CONTROL_BUSY;
        uint32_t time_start = hwtimer_get_time(t);

        // Write the data
        do {
            ret = adsp_write_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
            // Assert if operation is taking too long
            if (hwtimer_get_time(t) > time_start + CONTROL_COMMAND_TIMEOUT_TICKS) {
                xassert(0 && "Timer expired while writing control command");
            }
        }while(ret == ADSP_CONTROL_BUSY);
        xassert(ret == ADSP_CONTROL_SUCCESS);

        memset(cmd.payload, 0, cmd.payload_len);

        // Add a delay of 1 ms to address the bug described in https://xmosjira.atlassian.net/browse/LCD-257
        hwtimer_t t_delay = hwtimer_alloc();
        hwtimer_delay(t_delay, 100);
        hwtimer_free(t_delay);

        // Read back the written data
        ret = ADSP_CONTROL_BUSY;
        time_start = hwtimer_get_time(t);
        do {
            ret = adsp_read_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
            // Assert if operation is taking too long
            if (hwtimer_get_time(t) > time_start + CONTROL_COMMAND_TIMEOUT_TICKS) {
                xassert(0 && "Timer expired while reading control command");
            }
        }while(ret == ADSP_CONTROL_BUSY);

        xassert(ret == ADSP_CONTROL_SUCCESS);

        memcpy(values_read, cmd.payload, cmd.payload_len);

        // Check that read and written values match
        for(int i=0; i<cmd.payload_len; i++)
        {
            if(values_read[i] != values_write[i])
            {
                debug_printf("Command %d: mismatch at index %d. Expected %d, found %d\n", cmd.cmd_id, i, values_write[i], values_read[i]);
                xassert(0);
            }
        }
    }
    hwtimer_free(t);
}

#endif
