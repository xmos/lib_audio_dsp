// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#ifndef SPECIAL_COMMANDS_H_
#define SPECIAL_COMMANDS_H_

#include "command.hpp"

#define INVALID_INSTANCE_ID (0xff)

/**
 * @brief Return the absolute path to the command map file
 *
 * If no command map path is given in the CLI argument lists,
 * the default location will be used.
 *
 * @note Will decrement argc, if option is present
 */
std::string get_cmd_map_abs_path(int * argc, char ** argv);

/**
 * @brief Look up the string in the option list.
 *
 * If the string is not found, will suggest a possible match and exit.
 *
 * @param str   String sequence to look up
 * @note Function is case insensitive
 */
opt_t * option_lookup(const std::string str);

/**
 * @brief Gets device driver name to load by looking for --use
 *
 * @note Will decrement argc, if option is present
 */
std::string get_device_lib_name(int * argc, char ** argv);

/**
 * @brief Gets instance ID to use by looking for --instance-id
 *
 * @note Will decrement argc, if option is present
 */
uint8_t get_instance_id(int *argc, char **argv);

/** @brief Print application help menu */
control_ret_t print_help_menu();

/**
 * @brief Print command list loaded from the command_map
 *
 * @note Commands starting with SPECIAL_CMD_ and TEST_ will not be printed
 */
control_ret_t print_command_list();

std::vector<std::string> get_device_host_arg(int * argc, char ** argv, std::string lib_name);

#endif
