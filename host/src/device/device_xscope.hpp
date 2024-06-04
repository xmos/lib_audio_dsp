#ifndef DEVICE_XSCOPE_CLASS_H_
#define DEVICE_XSCOPE_CLASS_H_

#include "device.hpp"

class XSCOPE_Device : public Device
{
    public :
    XSCOPE_Device(int *device_info, std::string port_num)
    : Device(device_info)
    , port_num(port_num)
    {
    }

    public:
        control_ret_t device_init();
        std::string port_num;
};
#endif
