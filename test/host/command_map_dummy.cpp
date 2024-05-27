// Copyright 2023 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#include <stdlib.h>
#include <inttypes.h>
#include "device_control_shared.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <map>

/** @brief Enum for read/write command types */
enum cmd_rw_t {CMD_RO, CMD_WO, CMD_RW};

/**
 * @brief Enum for supported param types
 *
 * @note Add new cmd_param_type's to the end of the list.
 * @note TYPE_CHAR can only be READ ONLY.
 */
enum cmd_param_type_t {TYPE_CHAR, TYPE_UINT8, TYPE_INT32, TYPE_FLOAT, TYPE_UINT32, TYPE_RADIANS};

union cmd_param_t {uint8_t ui8; int32_t i32; float f; uint32_t ui32;};

/** @brief Command configuration structure
 *
 * @note cmd_name has to be upper case
 */
struct cmd_t
{
    /** Command resource ID */
    control_resid_t res_id;
    /** Command name */
    std::string cmd_name;
    /** Command value type */
    cmd_param_type_t type;
    /** Command ID */
    control_cmd_t cmd_id;
    /** Command read/write type */
    cmd_rw_t rw;
    /** Number of values the command reads/writes */
    unsigned num_values;
    /** Command info */
    std::string info;
    /** Command visibility status */
    bool hidden_cmd;
};

/** 
 * @brief Structure that keeps information needed to range check a single value
 * 
 * @note All values are inclusive
 */
struct val_range_t 
{
    /** Pointer to the array of intervals */
    cmd_param_t * ranges;
    /** Number of intervals to range check */
    size_t num_intervals;
};

#define PI_VALUE  3.14159265358979323846f

static cmd_t commands[] = {
                        {0, "CMD_FLOAT",   TYPE_FLOAT,   0,  CMD_RW, 20, "This is a test command for testing multiple float reads and writes. Need to keep command descriprion large to test -l option.",   false  },
                        {0, "CMD_INT32",   TYPE_INT32,   1,  CMD_RW, 20, "This is a test command for testing multiple int32 reads and writes. Need to keep command descriprion large to test -l option.",   false  },
                        {0, "CMD_UINT32",  TYPE_UINT32,  2,  CMD_RW, 20, "This is a test command for testing multiple uint32 reads and writes. Need to keep command descriprion large to test -l option.",  false  },
                        {0, "CMD_RADS",    TYPE_RADIANS, 3,  CMD_RW, 20, "This is a test command for testing multiple radians reads and writes. Need to keep command descriprion large to test -l option.", false  },
                        {0, "CMD_UINT8",   TYPE_UINT8,   4,  CMD_RW, 20, "This is a test command for testing multiple uint8 reads and writes. Need to keep command descriprion large to test -l option.",   false  },
                        {0, "CMD_CHAR",    TYPE_CHAR,    5,  CMD_RO, 20, "This is a test command for testing multiple char reads and writes. Need to keep command descriprion large to test -l option.",    false  },
                        {0, "CMD_HIDDEN",  TYPE_UINT8,   6,  CMD_RW, 20, "This command is suppossed to be hidden and not show up wehn using -l or -d",                                                      true   },
                        {0, "CMD_SMALL",   TYPE_INT32,   7,  CMD_RW, 3,  "This is a small command for testing -e option",                                                                                   false  },
                        {0, "RANGE_TEST0", TYPE_INT32,   8,  CMD_RW, 1,  "This command is used for the range check test",                                                                                   false  },
                        {0, "RANGE_TEST1", TYPE_FLOAT,   9,  CMD_RW, 3,  "This command is used for the range check test",                                                                                   false  },
                        {0, "RANGE_TEST2", TYPE_UINT8,   10, CMD_RW, 2,  "This command is used for the range check test",                                                                                   false  },
                        {0, "RANGE_TEST3", TYPE_UINT32,  11, CMD_RW, 3,  "This command is used for the range check test",                                                                                   false  }
};
static size_t num_commands = std::end(commands) - std::begin(commands);

extern "C"
size_t get_cmd_index(const std::string cmd_name)
{
    size_t index = UINT32_MAX;
    for(size_t i = 0; i < num_commands; i++)
    {
        cmd_t * cmd = &commands[i];
        if (cmd_name == cmd->cmd_name)
        {
            index = i;
            break;
        }
    }
    return index;
}

extern "C"
std::string get_cmd_name(const size_t index)
{
    return commands[index].cmd_name;
}

extern "C"
void get_cmd_id_info(control_resid_t * res_id, control_cmd_t * cmd_id, const size_t index)
{
    *res_id = commands[index].res_id;
    *cmd_id = commands[index].cmd_id;
}

extern "C"
void get_cmd_val_info(cmd_param_type_t * type, cmd_rw_t * rw, unsigned * num_vals, const size_t index)
{

    *type = commands[index].type;
    *rw = commands[index].rw;
    *num_vals = commands[index].num_values;
}

extern "C"
std::string get_cmd_info(const size_t index)
{
    return commands[index].info;
}

extern "C"
bool get_cmd_hidden(const size_t index)
{
    return commands[index].hidden_cmd;
}

extern "C"
uint32_t get_num_commands()
{
    return num_commands;
}

static const int dummy_info = 0x74736574;    // 'test' in ascii

extern "C"
const int * get_info_i2c()
{
    return &dummy_info;
}

void print_arg_local(const cmd_param_type_t type, const cmd_param_t val)
{
    switch(type)
    {
    case TYPE_CHAR:
        std::cout << static_cast<char>(val.ui8);
        break;
    case TYPE_UINT8:
        std::cout << static_cast<int>(val.ui8) << " ";
        break;
    case TYPE_FLOAT:
        std::cout << std::setprecision(7) << val.f << " ";
        break;
    case TYPE_RADIANS:
        std::cout << std::setprecision(5) << std::fixed << val.f << std::setprecision(2) << std::fixed << " (" << val.f  * 180.0f / PI_VALUE << " deg)" << " ";
        break;
    case TYPE_INT32:
        std::cout << val.i32 << " ";
        break;
    case TYPE_UINT32:
        std::cout << val.ui32 << " ";
        break;
    default:
        std::cerr << "Unsupported parameter type" << std::endl;
        exit(-1);
    }
}

extern "C"
void super_print_arg(const std::string cmd_name, cmd_param_t *values)
{
    cmd_t * cmd = &commands[get_cmd_index(cmd_name)];
    std::cout << cmd->cmd_name << " ";
    for(unsigned i = 0; i < cmd->num_values; i++)
    {
        print_arg_local(cmd->type, values[i]);
    }
    std::cout << std::endl;
}

void print_arg_stderr(const cmd_param_type_t type, const cmd_param_t val)
{
    switch(type)
    {
    case TYPE_CHAR:
        std::cerr << static_cast<char>(val.ui8);
        break;
    case TYPE_UINT8:
        std::cerr << static_cast<int>(val.ui8);
        break;
    case TYPE_RADIANS:
    case TYPE_FLOAT:
        std::cerr << std::setprecision(7) << val.f;
        break;
    case TYPE_INT32:
        std::cerr << val.i32;
        break;
    case TYPE_UINT32:
        std::cerr << val.ui32;
        break;
    default:
        std::cerr << "Unsupported parameter type" << std::endl;
        exit(-1);
    }
}

int compare_lim(const cmd_param_type_t type, const cmd_param_t val, const cmd_param_t * range)
{
    switch (type)
    {
    case TYPE_CHAR:
        std::cerr << "TYPE_CHAR commands can only be READ_ONLY" << std::endl;
        exit(-1);
    case TYPE_UINT8:
    {
        return ((val.ui8 >= range[0].ui8) && (val.ui8 <= range[1].ui8)) ? 1 : 0;
    }
    case TYPE_INT32:
    {
        return ((val.i32 >= range[0].i32) && (val.i32 <= range[1].i32)) ? 1 : 0;

    }
    case TYPE_UINT32:
    {
        std::cerr << "min " << range[0].ui32 << " max " << range[1].ui32 << " val " << val.ui32 << std::endl;
        return ((val.ui32 >= range[0].ui32) && (val.ui32 <= range[1].ui32)) ? 1 : 0;
    }
    case TYPE_FLOAT:
    case TYPE_RADIANS:
    {
        return ((val.f >= range[0].f) && (val.f <= range[1].f)) ? 1 : 0;
    }
    default:
        std::cerr << "Unsupported parameter type" << std::endl;
        exit(-1);
    }
}

void check_range_one(const val_range_t * range_info, const cmd_param_type_t cmd_type, const cmd_param_t param)
{
    size_t range_ind = 0;
    int ret = 0;
    for(size_t num_int = 0; num_int < range_info->num_intervals; num_int++)
    {

        ret = compare_lim(cmd_type, param, &range_info->ranges[num_int * 2]);
        if(ret == 1){break;}
    }
    if(ret == 0)
    {
        std::cerr << "Value ";
        print_arg_stderr(cmd_type, param);
        std::cerr << " must fall within the given range(s): ";

        for(size_t num_int = 0; num_int < range_info->num_intervals; num_int++)
        {
            std::cerr << "[";
            print_arg_stderr(cmd_type, range_info->ranges[num_int * 2]);
            std::cerr  << ", ";
            print_arg_stderr(cmd_type, range_info->ranges[num_int * 2 + 1]);
            std::cerr << "] ";
        }
        std::cerr << std::endl;
        exit(-1);
    }
}

cmd_param_t range0[4] = {0};
val_range_t val_range0[1] = {range0, 2};

cmd_param_t range1_0[2] = {0};
cmd_param_t range1_1[2] = {0};
cmd_param_t range1_2[2] = {0};
val_range_t val_range1[3] = {
                                {range1_0, 1},
                                {range1_1, 1},
                                {range1_2, 1},
};

cmd_param_t range2_0[2] = {0};
cmd_param_t range2_1[4] = {0};
val_range_t val_range2[2] = {
                                {range2_0, 1},
                                {range2_1, 2},
};

cmd_param_t range3_0[2] = {0};
cmd_param_t range3_2[2] = {0};
val_range_t val_range3[3] = {
                                {range3_0, 1},
                                {nullptr,  0},
                                {range3_2, 1},
};

std::map<std::string, val_range_t *> test_map{
    {
        "RANGE_TEST0", val_range0
    },
    {
        "RANGE_TEST1", val_range1
    },
    {
        "RANGE_TEST2", val_range2
    },
    {
        "RANGE_TEST3", val_range3
    },
};

extern "C"
void check_range(const std::string cmd_name, const cmd_param_t * vals)
{
    cmd_t * cmd = &commands[get_cmd_index(cmd_name)];

    if(cmd->cmd_name == "RANGE_TEST0")
    {
        range0[0].i32 = 0;
        range0[1].i32 = 3;
        range0[2].i32 = 5;
        range0[3].i32 = 5;
    }
    if(cmd->cmd_name == "RANGE_TEST1")
    {
        range1_0[0].f = 0.0;
        range1_0[1].f = 1.0;
        range1_1[0].f = 0.5;
        range1_1[1].f = 1.0;
        range1_2[0].f = 0.0;
        range1_2[1].f = 0.5;
    }
    if(cmd->cmd_name == "RANGE_TEST2")
    {
        range2_0[0].ui8 = 0;
        range2_0[1].ui8 = 3;
        range2_1[0].ui8 = 0;
        range2_1[1].ui8 = 6;
        range2_1[2].ui8 = 10;
        range2_1[3].ui8 = 14;
    }
    if(cmd->cmd_name == "RANGE_TEST3")
    {
        range3_0[0].ui32 = 0;
        range3_0[1].ui32 = 255;
        range3_2[0].ui32 = 0;
        range3_2[1].ui32 = 255;
    }
    if(test_map.find(cmd->cmd_name) != test_map.end())
    {
        val_range_t * val_range_ptr;
        val_range_ptr = test_map.find(cmd->cmd_name)->second;
        for(size_t i = 0; i < cmd->num_values; i++)
        {
            if(val_range_ptr[i].num_intervals == 0){continue;}
            check_range_one(&val_range_ptr[i], cmd->type, vals[i]);
        }
    }
}
