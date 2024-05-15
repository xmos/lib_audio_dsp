#include "print.h"
#include "adsp_control.h"
#include "adsp_instance_id_auto.h"
#include "xcore/hwtimer.h"
//#include "cmds.h"
//#include "cmd_offsets.h"

// The structs and enums before are used by host_cmd_map.h
enum cmd_param_type_t {TYPE_CHAR, TYPE_UINT8, TYPE_INT32, TYPE_FLOAT, TYPE_UINT32, TYPE_RADIANS};
/** @brief Enum for read/write command types */
enum cmd_rw_t {CMD_READ_ONLY, CMD_WRITE_ONLY, CMD_READ_WRITE};

typedef struct cmd_t
{
    /** Command resource ID */
    uint8_t res_id;
    /** Command name */
    char cmd_name[1000];
    /** Command value type */
    enum cmd_param_type_t type;
    /** Command ID */
    uint8_t cmd_id;
    /** Command read/write type */
    enum cmd_rw_t rw;
    /** Number of values the command reads/writes */
    unsigned num_values;
    /** Command info */
    char info[1000];
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
    char cmd_name[1000];
    uint8_t payload[1000];
}control_data_t;

#include "control_test_params.h"
//static limiter_rms_config_t config2 = { .attack_alpha = 89478485, .release_alpha = 894785, .threshold = 33713969 };
// hex(89478485) = '0x5555555'
// hex(829249) = '0xca741'
// hex(33713969) = '0x2026f31

uint8_t find_config_idx(char* cmd_name) {
    uint8_t config_size = sizeof(control_config) / sizeof(control_data_t);
    for (int i=0; i<config_size; i++) {
        if (strcmp(control_config[i].cmd_name,cmd_name) == 0) {
            return i;
        }
    }
    return 0xFF;
}
//limiter_rms_config_t config2 = { .attack_alpha = 89478485, .release_alpha = 894785, .threshold = 33713969 };

#include "print.h"
static void send_control_cmds(adsp_pipeline_t * m_dsp, chanend_t c_control) {
    adsp_stage_control_cmd_t cmd;
    int8_t payload_buf[256];
    cmd.instance_id = stage_test_stage_index;
    for (int cmd_idx = 0; cmd_idx<get_cmds_num(); cmd_idx++)
    {
        if (strncmp(stage_name, commands[cmd_idx].cmd_name, strlen(stage_name)) == 0) {

            cmd.cmd_id = commands[cmd_idx].cmd_id;
            cmd.payload_len = commands[cmd_idx].num_values * sizeof(int32_t);
            cmd.payload = payload_buf;
            memset(cmd.payload, 0, cmd.payload_len);
            //config2 = { .attack_alpha = 89478485, .release_alpha = 894785, .threshold = 33713969 };
            //int NUM_VALUES_LIMITER_RMS_ATTACK_ALPHA = cmd.payload_len / 4;
            //#define NUM_VALUES_LIMITER_RMS_ATTACK_ALPHA 1
            // Write control command to the stage
            uint8_t config_idx = find_config_idx(commands[cmd_idx].cmd_name);

            if (config_idx != 0xFF) {
                memcpy(cmd.payload, control_config[config_idx].payload, cmd.payload_len);
            }
            uint8_t values_write[1000];
            memcpy(values_write, cmd.payload, cmd.payload_len);
            adsp_control_status_t ret = ADSP_CONTROL_BUSY;
            do {
                ret = adsp_write_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
            }while(ret == ADSP_CONTROL_BUSY);
            assert(ret == ADSP_CONTROL_SUCCESS);
            memset(cmd.payload, 0, cmd.payload_len);
            hwtimer_t t = hwtimer_alloc(); hwtimer_delay(t, 100); //100us to allow command to be written

            // Read back the written data
            ret = ADSP_CONTROL_BUSY;
            do {
                ret = adsp_read_module_config(m_dsp->modules, m_dsp->n_modules, &cmd);
            }while(ret == ADSP_CONTROL_BUSY);

            assert(ret == ADSP_CONTROL_SUCCESS);

            uint8_t values_read[1000];
            memcpy(values_read, cmd.payload, cmd.payload_len);
            // Check that the configurable values are correct,
            // the other commands may be overwritten by the stage
            if (config_idx != 0xFF) {
                for(int i=0; i<cmd.payload_len; i++)
                {
                    if(values_read[i] != values_write[i])
                    {
                        printf("Command %d: mismatch at index %d. Expected %d, found %d\n", cmd.cmd_id, i, values_write[i], values_read[i]);
                        assert(0);
                    }
                }
            }
        }
    }
}