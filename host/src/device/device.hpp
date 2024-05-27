// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#ifndef DEVICE_CLASS_H_
#define DEVICE_CLASS_H_

extern "C"
#include "device_control_shared.h"
#include <memory>
#include <iostream>
#include <vector>

/**
 * @brief Class for interfacing device_contol
 */
class Device
{
    private:

        /** @brief Information to intialise the device */
        int * device_info;

    protected:

        /**
         * @brief State of the device.
         *
         * Used to prevent multiple initialisations / cleanups.
         */
        bool device_initialised = false;

    public:

        /**
         * @brief Construct a new Device object
         *
         * @param device_info   Pointer to the information to initialise a device
         */
        Device(int * device_info);

        /** @brief Initialise a host (master) interface */
        virtual control_ret_t device_init();

        /**
         * @brief Request to read from a controllable resource inside the device
         *
         * @param res_id        Resource ID
         * @param cmd_id        Command ID
         * @param payload       Array of bytes which constitutes the data payload
         * @param payload_len   Size of the payload in bytes
         */
        virtual control_ret_t device_get(control_resid_t res_id, control_cmd_t cmd_id, uint8_t payload[], size_t payload_len);

        /**
         * @brief Request to write to a controllable resource inside the device
         *
         * @param res_id        Resource ID
         * @param cmd_id        Command ID
         * @param payload       Array of bytes which constitutes the data payload
         * @param payload_len   Size of the payload in bytes
         */
        virtual control_ret_t device_set(control_resid_t res_id, control_cmd_t cmd_id, const uint8_t payload[], size_t payload_len);

        /**
         * @brief Destroy the Device object.
         *
         * This will shut down the host interface connection.
         */
        virtual ~Device();
};

extern "C"
/**
 * @brief Returns a pointer to the Device class object
 *
 * @param info      Pointer to the information to initialise a device
 */
Device * make_Dev(int * info, std::vector<std::string> vec);

#endif
