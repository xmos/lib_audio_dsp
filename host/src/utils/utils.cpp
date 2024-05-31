// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#define _USE_MATH_DEFINES  // required for windows in order for cmath to expose M_PI

#include "utils.hpp"
#include <vector>
#include <iomanip>
#include <cmath>
#include "control_ret_str_map.h"

using namespace std;

static cmd_index_fptr get_cmd_index = nullptr;
static cmd_name_fptr get_cmd_name = nullptr;
static cmd_id_info_fptr get_cmd_id_info = nullptr;
static cmd_val_info_fptr get_cmd_val_info = nullptr;
static cmd_info_fptr get_cmd_info = nullptr;

size_t num_commands = 0;

string to_upper(string str)
{
    for(unsigned i = 0; i < str.length(); i++)
    {
        str[i] = toupper(str[i]);
    }
    return str;
}

string to_lower(string str)
{
    for(unsigned i = 0; i < str.length(); i++)
    {
        str[i] = tolower(str[i]);
    }
    return str;
}

int * get_device_init_info(dl_handle_t handle, string lib_name)
{
    string symbol;
    if(lib_name == device_usb_dl_name)
    {
        symbol = "get_info_usb";
    }
    else if(lib_name == device_xscope_dl_name)
    {
        symbol = "get_info_xscope";
    }
    else
    {
        cerr << "Not a valid device dl name " << lib_name << endl;
        exit(HOST_APP_ERROR);
    }
    device_info_fptr get_device_info = get_device_info_fptr(handle, symbol);

    return get_device_info();
}

dl_handle_t load_command_map_dll(const string cmd_map_abs_path)
{
    dl_handle_t handle = get_dynamic_lib(cmd_map_abs_path);

    num_cmd_fptr get_num_commands = get_num_cmd_fptr(handle);
    num_commands = get_num_commands();

    get_cmd_index = get_cmd_index_fptr(handle);
    get_cmd_name = get_cmd_name_fptr(handle);
    get_cmd_id_info = get_cmd_id_info_fptr(handle);
    get_cmd_val_info = get_cmd_val_info_fptr(handle);
    get_cmd_info = get_cmd_info_fptr(handle);

    return handle;
}

bool check_if_cmd_exists(const string cmd_name)
{
    const string up_str = to_upper(cmd_name);
    size_t index = get_cmd_index(up_str);
    if(index == UINT32_MAX)
    {
        return false;
    }
    return true;
}

void init_cmd(cmd_t * cmd, const std::string cmd_name, size_t index)
{
    const string up_str = to_upper(cmd_name);

    if(index == UINT32_MAX)
    {
        index = get_cmd_index(up_str);
        /*if(index == UINT32_MAX)
        {
            calc_Levenshtein_and_error(up_str);
        }*/
        cmd->cmd_name = up_str;
    }
    else
    {
        cmd->cmd_name = get_cmd_name(index);
    }

    get_cmd_id_info(&cmd->res_id, &cmd->cmd_id, index);
    get_cmd_val_info(&cmd->type, &cmd->rw, &cmd->num_values, index);
    cmd->info = get_cmd_info(index);
}

size_t argv_option_lookup(int argc, char ** argv, opt_t * opt_lookup)
{
    for(int i = 1; i < argc; i++)
    {
        string cmd_arg = to_lower(argv[i]);
        if((cmd_arg == opt_lookup->long_name) || (cmd_arg == opt_lookup->short_name))
        {
            return i;
        }
    }
    return 0;
}

void remove_opt(int * argc, char ** argv, size_t ind, size_t num)
{
    for(size_t i = 0; i < * argc - ind - num; i++)
    {
        argv[ind + i] = argv[ind + num + i];
    }
    * argc -= num;
    if(* argc == 1)
    {
        cout << "Use --help to get the list of options for this application." << endl
        << "Or use --list-commands to print the list of commands and their info." << endl;
        exit(0);
    }
}

void check_cmd_error(string cmd_name, string rw, control_ret_t ret)
{
    rw[0] = toupper(rw[0]);
    if(ret != CONTROL_SUCCESS)
    {
        cerr << rw << " command " << cmd_name << " returned control_ret_t error " << static_cast<int>(ret) << ", " << control_ret_str_map[ret] << endl;
        exit(ret);
    }
}

string command_rw_type_name(const cmd_rw_t rw)
{
    string tstr;

    switch(rw){
    case CMD_RO:
        tstr = "READ ONLY";
        break;

    case CMD_WO:
        tstr = "WRITE ONLY";
        break;

    case CMD_RW:
        tstr = "READ/WRITE";
        break;

    default:
        cerr << "Unsupported read/write type" << endl;
        exit(HOST_APP_ERROR);
    }

    return tstr;
}

control_ret_t check_num_args(const cmd_t * cmd, const size_t args_left)
{
    if((cmd->rw == CMD_RO) && (args_left != 0))
    {
        cerr << "Command: " << cmd->cmd_name << " is read-only, so it does not require any arguments." << endl;
        exit(HOST_APP_ERROR);
    }
    else if ((cmd->rw == CMD_WO) && (args_left != cmd->num_values))
    {
        cerr << "Command: " << cmd->cmd_name << " is write-only and"
        << " expects " << cmd->num_values << " argument(s), " << endl
        << args_left << " are given." << endl;
        exit(HOST_APP_ERROR);
    }
    else if ((cmd->rw == CMD_RW) && (args_left != 0) && (args_left != cmd->num_values))
    {
        cerr << "Command: " << cmd->cmd_name << " is a read/write command." << endl
        << "If you want to read do not give any arguments to this command." << endl
        << "If you want to write give " << cmd->num_values << " argument(s) to this command, "
        << args_left << " are given." << endl;
        exit(HOST_APP_ERROR);
    }
    return CONTROL_SUCCESS;
}

cmd_param_t command_bytes_to_value(const cmd_param_type_t type, const uint8_t * data, unsigned index)
{
    cmd_param_t value;
    size_t size_bytes = get_num_bytes_from_type(type);

    switch(size_bytes)
    {
    case 1:
        memcpy(&value.ui8, data + index * size_bytes, size_bytes);
        break;
    case 4:
        memcpy(&value.i32, data + index * size_bytes, size_bytes);
        break;
    default:
        cerr << "Unsupported parameter type" << endl;
        exit(HOST_APP_ERROR);
    }

    return value;
}

void command_bytes_from_value(const cmd_param_type_t type, uint8_t * data, unsigned index, const cmd_param_t value)
{
    size_t num_bytes = get_num_bytes_from_type(type);
    switch(num_bytes)
    {
    case 1:
        memcpy(data + index * num_bytes, &value.ui8, num_bytes);
        break;
    case 4:
        memcpy(data + index * num_bytes, &value.i32, num_bytes);
        break;
    default:
        cerr << "Unsupported parameter type" << endl;
        exit(HOST_APP_ERROR);
    }
}

void print_read_result(cmd_t cmd, cmd_param_t *cmd_values)
{
    for(unsigned i=0; i<cmd.num_values; i++)
    {
        cmd_param_type_t type = cmd.type;
        cmd_param_t val = cmd_values[i];
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
    std::cout << std::endl;
}
