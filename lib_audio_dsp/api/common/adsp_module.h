#ifndef DSPT_MODULE_H
#define DSPT_MODULE_H

#include <stdint.h>
#include <stdbool.h>
#include <xmath/xmath.h>

#define DSP_INPUT_CHANNELS (4)  // For the 2ch USB + 2ch I2S config for now. TODO: Fix this
#define DSP_OUTPUT_CHANNELS (DSP_INPUT_CHANNELS)

typedef enum
{
    parametric_eq = 0,
    agc,
    biquad,
    num_dsp_modules
}all_dsp_modules_t;

typedef enum
{
    config_read_pending,    // Control waiting to read the updated config from DSP
    config_write_pending,   // Config written by control and waiting for DSP to update
    config_read_updated,    // Config updated with the latest in order to service a read command
    config_none_pending     // All done. Control and DSP not waiting on anything.
}config_rw_state_t;

// Control related information shared between control thread and DSP
typedef struct
{
    void *config;
    uint32_t id;    // Unique module identifier assigned by the host
    uint32_t num_control_commands;
    all_dsp_modules_t module_type;
    uint8_t cmd_id;
    config_rw_state_t config_rw_state;
}module_control_t;

#define DSP_MODULE_PROCESS_ATTR  __attribute__((fptrgroup("dsp_module_process_fptr_grp")))
typedef void (*dsp_module_process)(int32_t *input, int32_t *output, void *state, module_control_t *control);

typedef struct
{
    void *state;    // Pointer to the module's state memory
    DSP_MODULE_PROCESS_ATTR dsp_module_process process_sample;  // Pointer to the module's process_sample() function
    // For control
    module_control_t control;
}module_instance_t;


#define DSP_MODULE_INIT_ATTR  __attribute__((fptrgroup("dsp_module_init_fptr_grp")))
typedef module_instance_t* (*dsp_module_init)(uint8_t id);


typedef struct
{
    uint8_t instance_id;
    DSP_MODULE_INIT_ATTR dsp_module_init module_init_function;  // Pointer to the module's init function
}module_info_t;


#endif
