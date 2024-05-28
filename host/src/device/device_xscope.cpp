// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "device.hpp"
#include "device_control_host.h"
#include "device_xscope.hpp"

using namespace std;

Device::Device(int * info)
{
    device_info = info;
}

control_ret_t Device::device_init()
{
    control_ret_t ret = CONTROL_SUCCESS;
    return ret;
}


control_ret_t XSCOPE_Device::device_init()
{
    control_ret_t ret = CONTROL_SUCCESS;
    if(!device_initialised)
    {
        ret = control_init_xscope("localhost", port_num.c_str());
        device_initialised = true;
    }
    return ret;
}

control_ret_t Device::device_get(control_resid_t res_id, control_cmd_t cmd_id, uint8_t payload[], size_t payload_len)
{
    control_ret_t ret = control_read_command(res_id, cmd_id, payload, payload_len);
    return ret;
}

control_ret_t Device::device_set(control_resid_t res_id, control_cmd_t cmd_id, const uint8_t payload[], size_t payload_len)
{
    control_ret_t ret = control_write_command(res_id, cmd_id, payload, payload_len);
    return ret;
}

Device::~Device()
{
    if(device_initialised)
    {
        control_cleanup_xscope();
        device_initialised = false;
    }
}

extern "C"
Device * make_Dev(int * info, vector<string> vec)
{
    static XSCOPE_Device dev_obj(info, vec[0]);
    return &dev_obj;
}
