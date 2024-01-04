#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <xcore/assert.h>
#include "debug_print.h"
#include "adsp_module.h"
#include "adsp_control.h"


typedef struct {
    uint32_t cmd_id; // CmdID
    uint32_t offset;    // offset
    uint32_t size;      //size
}module_config_offsets_t;
#include "cmd_offsets.h"    // Autogenerated

static module_instance_t* get_module_instance(module_instance_t **modules, uint32_t res_id, size_t num_modules)
{
    //printf("res id = %d\n", res_id);
    for(int i=0; i<num_modules; i++)
    {
        if(modules[i]->control.id == res_id)
        {
            return modules[i];
        }
    }
    printf("ERROR: Cannot find a module for the instance-id %lu\n", res_id);
    xassert(0);
    return NULL;
}

static void get_control_cmd_config_offset(module_instance_t *module, uint8_t cmd_id, uint32_t *offset, uint32_t *size)
{
    //printf("cmd id = %d\n", cmd_id);
    all_dsp_modules_t module_type = module->control.module_type;
    module_config_offsets_t *config_offsets = ptr_module_offsets[module_type];

    for(int i=0; i<module->control.num_control_commands; i++)
    {
        if(cmd_id == (uint8_t)config_offsets[i].cmd_id)
        {
            *offset = config_offsets[i].offset;
            *size = config_offsets[i].size;
            return;
        }
    }
    printf("ERROR: cmd_id %d not found in module_type %d\n", cmd_id, module_type);
    xassert(0);
    return;
}

// Read a module instance's config structure for a given command ID
adsp_control_status_t adsp_read_module_config(module_instance_t** modules, // Array of module instance pointers
                                            size_t num_modules, // Total number of modules
                                            adsp_stage_control_cmd_t *cmd
                                        )
{
    module_instance_t *module = get_module_instance(modules, cmd->instance_id, num_modules);
    uint32_t offset, size;
    // Get offset into the module's config structure for this command
    get_control_cmd_config_offset(module, cmd->cmd_id, &offset, &size);
    if(size != cmd->payload_len)
    {
        debug_printf("ERROR: payload_len mismatch. Expected %lu, but received %u\n", size, cmd->payload_len);
        xassert(0);
    }
    config_rw_state_t config_state = module->control.config_rw_state;
    if((config_state == config_none_pending) || (config_state == config_read_pending)) // No command pending or read pending
    {
        if(config_state == config_none_pending)
        {
            // Inform the module of the read so it can update config with the latest data
            module->control.cmd_id = cmd->cmd_id;
            module->control.config_rw_state = config_read_pending;
        }
        // Return RETRY as status
        return ADSP_CONTROL_BUSY;
    }
    else if(config_state == config_read_updated)
    {
        // Confirm same cmd_id
        xassert(module->control.cmd_id == cmd->cmd_id);
        // Update payload
        memcpy((uint8_t*)&cmd->payload[0], (uint8_t*)module->control.config + offset, size);
        module->control.config_rw_state = config_none_pending;
        return ADSP_CONTROL_SUCCESS;
    }
    // Should never come here
    debug_printf("adsp_read_module_config(): Unexpected config state %d\n", config_state);
    xassert(0);
    return ADSP_CONTROL_ERROR;
}


// Write to a module instance's config structure for a given command ID
adsp_control_status_t adsp_write_module_config(module_instance_t** modules, // Array of module instance pointers
                                            size_t num_modules, // Total number of modules
                                            adsp_stage_control_cmd_t *cmd
                                        )
{
    module_instance_t *module = get_module_instance(modules, cmd->instance_id, num_modules);
    uint32_t offset, size;
    // Get offset into the module's config structure for this command
    get_control_cmd_config_offset(module, cmd->cmd_id, &offset, &size);
    if(size != cmd->payload_len)
    {
        debug_printf("ERROR: payload_len mismatch. Expected %lu, but received %u\n", size, cmd->payload_len);
        xassert(0);
    }

    config_rw_state_t config_state = module->control.config_rw_state;
    if(config_state == config_none_pending)
    {
        // Receive write payload
        memcpy((uint8_t*)module->control.config + offset, cmd->payload, cmd->payload_len);
        module->control.cmd_id = cmd->cmd_id;
        module->control.config_rw_state = config_write_pending;
        return ADSP_CONTROL_SUCCESS;
    }
    else
    {
        debug_printf("WARNING: Previous write to the config not applied by the module!! Ignoring write command.");
        return ADSP_CONTROL_BUSY;
    }
}
