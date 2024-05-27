// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#ifndef COMMAND_UTILS_H_
#define COMMAND_UTILS_H_

#include <cstring>
#include <cstdint>
#include "device.hpp"

#if defined(_WIN32)
#include <string>
#include <windows.h>
/** @brief Typedef for storing dynamically opened library  */
typedef HMODULE dl_handle_t;
#elif (defined(__APPLE__) || defined(__linux__))
/** @brief Typedef for storing dynamically opened library  */
typedef void * dl_handle_t;
#endif

#define HOST_APP_ERROR -1

/** @brief Enum for read/write command types */
enum cmd_rw_t {CMD_RO, CMD_WO, CMD_RW};

/**
 * @brief Enum for supported param types
 *
 * @note Add new cmd_param_type's to the end of the list.
 */
enum cmd_param_type_t {TYPE_UINT8, TYPE_INT32, TYPE_FLOAT, TYPE_UINT32};

/** @brief Union for supporting different command param types */
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

/** @brief Option configuration structure
 *
 * @note Option names have to be lower case
 */
struct opt_t
{
    /** Long option name */
    std::string long_name;
    /** Short option name */
    std::string short_name;
    /** Option info */
    std::string info;
};
/** @brief I2C device driver name */
const std::string device_i2c_dl_name = "device_i2c";

/** @brief SPI device driver name */
const std::string device_spi_dl_name = "device_spi";

/** @brief USB device driver name */
const std::string device_usb_dl_name = "device_usb";

/** @brief XSCOPE device driver name */
const std::string device_xscope_dl_name = "device_xscope";

/** @brief Default driver name to use */
const std::string default_driver_name = device_xscope_dl_name;

/** @brief Default command_map dl name to use */
const std::string default_command_map_name = "command_map";

/** @brief Current version of this application
 *
 * @note This will have to be manually changed after the release
 */
const std::string current_host_app_version = "0.1.0";

/** @brief Convert string to uper case */
std::string to_upper(std::string str);

/** @brief Convert string to lower case */
std::string to_lower(std::string str);

/**
 * @brief Get information to initialise a device
 *
 * @param handle    Pointer to the comamnd_map dl
 * @param lib_name  Device dl name
 */
int * get_device_init_info(dl_handle_t handle, std::string lib_name);

/** @brief Load the command_map shared object and get the cmd tools from it */
dl_handle_t load_command_map_dll(const std::string cmd_map_abs_path);

/** @brief Initialise cmd_t structure with either command name or it's index */
void init_cmd(cmd_t * cmd, const std::string cmd_name, size_t index = UINT32_MAX);

/** @brief Lookup option in argv */
size_t argv_option_lookup(int argc, char ** argv, opt_t * opt_lookup);

/** @brief Remove words from argv, decrement argc */
void remove_opt(int * argc, char ** argv, size_t ind, size_t num);

/**
 * @brief Convert relative path to working directory to absolute path
 *
 * @param rel_path Path of the file relative to the current working directory
 */
std::string convert_to_abs_path(const std::string rel_path);

/**
 * @brief Convert lib name into the path to the library
 *
 * @param lib_name Name of the library to load (without lib prefix)
 */
std::string get_dynamic_lib_path(const std::string lib_name);

/**
 * @brief Open the dynamic library
 *
 * @param lib_path Path to the library to load (without lib prefix)
 */
dl_handle_t get_dynamic_lib(const std::string lib_path);

/** uint32_t function pointer type */
using num_cmd_fptr = uint32_t (*)();

/** Function pointer for getting index if the command */
using cmd_index_fptr = size_t (*)(const std::string);

/** Function pointer for getting command name form index */
using cmd_name_fptr = std::string (*)(const size_t);

/** Function pointer for getting cmd id related info */
using cmd_id_info_fptr = void (*)(control_resid_t *, control_cmd_t *, const size_t);

/** Function pointer for getting cmd val related info */
using cmd_val_info_fptr = void (*)(cmd_param_type_t *, cmd_rw_t *, unsigned *, const size_t);

/** Function pointer for getting cmd info */
using cmd_info_fptr = std::string (*)(const size_t);

/** Function pointer that takes void * and returns Device */
using device_fptr = Device * (*)(int *, std::vector<std::string>);

/** Function pointer for getting the information to initialise a device */
using device_info_fptr = int * (*)();

/**
 * @brief Get the function pointer to get_num_commands()
 *
 * @param handle Pointer to the command_map shared object
 */
num_cmd_fptr get_num_cmd_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to get_cmd_name()
 *
 * @param handle Pointer to the command_map shared object
 */
cmd_name_fptr get_cmd_name_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to get_cmd_index()
 *
 * @param handle Pointer to the command_map shared object
 */
cmd_index_fptr get_cmd_index_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to get_cmd_id_info()
 *
 * @param handle Pointer to the command_map shared object
 */
cmd_id_info_fptr get_cmd_id_info_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to get_cmd_val_info()
 *
 * @param handle Pointer to the command_map shared object
 */
cmd_val_info_fptr get_cmd_val_info_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to get_cmd_info()
 *
 * @param handle Pointer to the command_map shared object
 */
cmd_info_fptr get_cmd_info_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to make_Dev()
 *
 * @param handle Pointer to the device shared object
 */
device_fptr get_device_fptr(dl_handle_t handle);

/**
 * @brief Get the function pointer to get_info_***()
 *
 * @param handle Pointer to the command_map shared object
 * @param symbol Name of the function to lookup
 */
device_info_fptr get_device_info_fptr(dl_handle_t handle, const std::string symbol);

/** @brief Get param type name string */
std::string command_param_type_name(const cmd_param_type_t type);

/** @brief Get read/write type string */
std::string command_rw_type_name(const cmd_rw_t rw);

/** @brief Check if the right number of arguments has been given for the command */
control_ret_t check_num_args(const cmd_t * cmd, const size_t args_left);

/** @brief Convert command line argument from string to cmd_param_t */
cmd_param_t cmd_arg_str_to_val(const cmd_param_type_t type, const char * str);

/** @brief Exit on control_ret_t error */
void check_cmd_error(std::string cmd_name, std::string rw, control_ret_t ret);

/** @brief Get number of bytes for the particular param type */
size_t get_num_bytes_from_type(const cmd_param_type_t type);

/** @brief Convert single value from bytes to cmd_param_t */
cmd_param_t command_bytes_to_value(const cmd_param_type_t type, const uint8_t * data, unsigned index);

/** @brief Convert single value from cmd_param_t to bytes */
void command_bytes_from_value(const cmd_param_type_t type, uint8_t * data, unsigned index, const cmd_param_t value);

/** @brief Find Levenshtein distance for approximate string matching */
//int Levenshtein_distance(const std::string source, const std::string target);

/** @brief Get current terminal width */
size_t get_term_width();

/** @brief Check if a given command is present in the command_map.
 * @return true if command present, false otherwise
*/
bool check_if_cmd_exists(const std::string cmd_name);

void print_read_result(cmd_t cmd, cmd_param_t *cmd_values);

#endif
