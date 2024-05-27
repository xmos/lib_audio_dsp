// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#ifndef COMMAND_CLASS_H_
#define COMMAND_CLASS_H_

#include "utils.hpp"

/**
 * @brief Class for executing a single command
 */
class Command
{
    private:

        /** @brief Pointer to the Device class object */
        Device * device;

        /** @brief Command information */
        cmd_t cmd;

        uint8_t instance_id;

    public:

        /**
         * @brief Construct a new Command object.
         *
         * Will initialise a host (master) interface.
         *
         * @param _dev          Pointer to the Device class object
         * @param _handle       Command map dl handle
         */
        Command(Device * _dev, dl_handle_t _handle, uint8_t _instance_id);

        /**
         * @brief Initialise command information
         *
         * @param cmd_name      The command name to be executed
         * @note This has to be used if using command_get() or command_set()
         */
        void init_cmd_info(const std::string cmd_name);

        /**
         * @brief Takes argv and executes a single command from it
         *
         * @param cmd_name      The command name to be executed
         * @param argv          Pointer to command line arguments
         * @param argc          Number of arguments in command line
         * @param arg_indx      Index of argv to look at
         */
        control_ret_t do_command(const std::string cmd_name, char ** argv, int argc, int arg_indx);

        /**
         * @brief Executes a single get comamnd
         *
         * @param values        Pointer to store values read from the device
         */
        control_ret_t command_get(cmd_param_t * values);

        /**
         * @brief Executes a single set command
         *
         * @param values        Pointer to store values to write to the device
         */
        control_ret_t command_set(const cmd_param_t * values);
};

#endif
