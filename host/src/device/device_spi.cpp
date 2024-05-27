// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#include "device.hpp"
#include "device_control_host.h"
#include "bcm2835.h"

using namespace std;

static constexpr long intertransaction_delay_ns = 0;

Device::Device(int* info)
{
    device_info = info;
}

control_ret_t Device::device_init()
{
    control_ret_t ret = CONTROL_SUCCESS;
    if(!device_initialised)
    {
        ret = control_init_spi_pi(static_cast<spi_mode_t>(device_info[0]),
                                  static_cast<bcm2835SPIClockDivider>(device_info[1]),
                                  intertransaction_delay_ns);
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
        control_cleanup_spi();
        device_initialised = false;
    }
}

extern "C"
Device * make_Dev(int * info, std::vector<std::string> vec)
{
    static Device dev_obj(info);
    return &dev_obj;
}
