#include <string.h>
#include "adsp_control.h"
#include "adsp_instance_id_auto.h"
#include "xcore/hwtimer.h"
#include <xcore/assert.h>
#include <debug_print.h>

#include "run_cmds.h"

#define CMD_NAME_MAX_SIZE 256
#define CMD_PAYLOAD_MAX_SIZE 256
#define CMD_INFO_MAX_SIZE 1024

// The structs and enums before are used by host_cmd_map.h
enum cmd_param_type_t {TYPE_CHAR, TYPE_UINT8, TYPE_INT32, TYPE_FLOAT, TYPE_UINT32, TYPE_RADIANS};
/** @brief Enum for read/write command types */
enum cmd_rw_t {CMD_READ_ONLY, CMD_WRITE_ONLY, CMD_READ_WRITE};

typedef struct cmd_t
{
    /** Command resource ID */
    uint8_t res_id;
    /** Command name */
    char cmd_name[CMD_NAME_MAX_SIZE];
    /** Command value type */
    enum cmd_param_type_t type;
    /** Command ID */
    uint8_t cmd_id;
    /** Command read/write type */
    enum cmd_rw_t rw;
    /** Number of values the command reads/writes */
    unsigned num_values;
    /** Command info */
    char info[CMD_INFO_MAX_SIZE];
    /** Command visibility status */
    bool hidden_cmd;
}cmd_t;

#include "host_cmd_map.h"

uint8_t get_cmds_num() {
    return sizeof(commands) / sizeof(cmd_t);
}

uint8_t find_cmd_idx(char* cmd_name) {
    for (int i=0; i<get_cmds_num(); i++) {
        if (strcmp(commands[i].cmd_name,cmd_name) == 0) {
            return i;
        }
    }
    xassert(0 && "Command not found");
    return 0xFF;
}

typedef struct control_data_t
{
    char cmd_name[CMD_NAME_MAX_SIZE];
    uint8_t payload[CMD_PAYLOAD_MAX_SIZE];
}control_data_t;

uint8_t get_value_size(enum cmd_param_type_t value_type) {
    switch (value_type) {
        case TYPE_CHAR:
        case TYPE_UINT8:
            return 1;
        case TYPE_INT32:
        case TYPE_FLOAT:
        case TYPE_UINT32:
        case TYPE_RADIANS:
            return 4;
        default:
            xassert(0 && "Invalid value type");
    }
}

#if SEND_CONTROL_COMMANDS
#include "control_test_params.h"

uint8_t find_config_idx(char* cmd_name) {
    uint8_t config_size = sizeof(control_config) / sizeof(control_data_t);
    for (int i=0; i<config_size; i++) {
        if (strcmp(control_config[i].cmd_name,cmd_name) == 0) {
            return i;
        }
    }
    return 0xFF;
}
#endif

void send_control_cmds(adsp_pipeline_t * m_dsp, chanend_t c_control) {
#if SEND_CONTROL_COMMANDS
    adsp_stage_control_cmd_t cmd;
    int8_t payload_buf[CMD_PAYLOAD_MAX_SIZE];
    cmd.instance_id = control_stage_index;
    for (int cmd_idx = 0; cmd_idx<get_cmds_num(); cmd_idx++)
    {
        if (strncmp(stage_name, commands[cmd_idx].cmd_name, strlen(stage_name)) == 0) {

            cmd.cmd_id = commands[cmd_idx].cmd_id;

            cmd.payload_len = commands[cmd_idx].num_values * get_value_size(commands[cmd_idx].type);
            cmd.payload = payload_buf;
            memset(cmd.payload, 0, cmd.payload_len);

            // Write control command to the stage
            uint8_t config_idx = find_config_idx(commands[cmd_idx].cmd_name);

            if (config_idx != 0xFF) {
                memcpy(cmd.payload, control_config[config_idx].payload, cmd.payload_len);
            }
            uint8_t values_write[CMD_PAYLOAD_MAX_SIZE];
            memcpy(values_write, cmd.payload, cmd.payload_len);
            adsp_control_status_t ret = ADSP_CONTROL_BUSY;
            do {
                ret = adsp_write_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
            }while(ret == ADSP_CONTROL_BUSY);
            xassert(ret == ADSP_CONTROL_SUCCESS);
            memset(cmd.payload, 0, cmd.payload_len);

            hwtimer_t t = hwtimer_alloc(); hwtimer_delay(t, 100); //100us to allow command to be written

            // Read back the written data
            ret = ADSP_CONTROL_BUSY;
            do {
                ret = adsp_read_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
            }while(ret == ADSP_CONTROL_BUSY);

            xassert(ret == ADSP_CONTROL_SUCCESS);

            uint8_t values_read[CMD_PAYLOAD_MAX_SIZE];
            memcpy(values_read, cmd.payload, cmd.payload_len);
            // Check that the configurable values are correct,
            // the other commands may differ as the stage can overwrite them
            if (config_idx != 0xFF) {
                for(int i=0; i<cmd.payload_len; i++)
                {
                    if(values_read[i] != values_write[i])
                    {
                        debug_printf("Command %d: mismatch at index %d. Expected %d, found %d\n", cmd.cmd_id, i, values_write[i], values_read[i]);
                        xassert(0);
                    }
                }
            }
        }
    }
#endif
}
