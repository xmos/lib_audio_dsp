// Copyright 2023 XMOS LIMITED.
// This Software is subject to the terms of the XCORE VocalFusion Licence.

#include "device.hpp"
#include <cstring>
#include <fstream>
#include <vector>

#define NUM_CHARS 20 // This number is defined in command_map_dummy.cpp

using namespace std;


uint8_t buffer[NUM_CHARS * sizeof(float)] = {0};
const string buf_filename = "test_buf.bin";
const size_t buff_size = end(buffer) - begin(buffer);
const char ch_ar[NUM_CHARS] = "Hello New World!\0\0\0";

Device::Device(int * info)
{
    device_info = info;
}

control_ret_t Device::device_init()
{
    control_ret_t ret = CONTROL_SUCCESS;
    if(!device_initialised)
    {
        char check[5] = {'0', '0', '0', '0', '\0'};
        memcpy(check, device_info, 4 * sizeof(char));
        string check_str = check;
        if(check_str != "test")
        {
            ret = CONTROL_REGISTRATION_FAILED;
        }
        device_initialised = true;
    }
    return ret;
}

control_ret_t Device::device_get(control_resid_t res_id, control_cmd_t cmd_id, uint8_t payload[], size_t payload_len)
{
    if(payload_len > (buff_size + 1)) // Max buffer len + 1 for the status
    {
        return CONTROL_DATA_LENGTH_ERROR;
    }
    payload[0] = 0;

    ifstream rf(buf_filename, ios::in | ios::binary);
    if(!rf)
    {
        cerr << "Could not open a file " << buf_filename << endl;
        return CONTROL_ERROR;
    }

    rf.seekg (0, rf.end);
    streamoff size = rf.tellg();
    rf.seekg (0, rf.beg);

    if(size != buff_size)
    {
        cerr << "Test buffer lengths don't match" << endl;
        return CONTROL_DATA_LENGTH_ERROR;
    }

    for(int i = 0; i < buff_size; i++)
    {
        rf.read(reinterpret_cast<char *>(&buffer[i]), sizeof(uint8_t));
    }

    rf.peek(); // doing peek here to look for a character beyond file size so it will set eof
    rf.close();

    if(!rf.eof() || rf.bad())
    {
        cerr << "Error occured while reading " << buf_filename << endl;
        exit(CONTROL_ERROR);
    }

    switch(cmd_id & 0x7F)
    {
        case 5:
            memcpy(&payload[1], ch_ar, payload_len);
            break;
        default:
            memcpy(&payload[1], buffer, payload_len);
    }
    return CONTROL_SUCCESS;
}

control_ret_t Device::device_set(control_resid_t res_id, control_cmd_t cmd_id, const uint8_t payload[], size_t payload_len)
{
    if(payload_len > buff_size) // Max buffer len
    {
        return CONTROL_DATA_LENGTH_ERROR;
    }

    memcpy(buffer, payload, payload_len);

    ofstream wf(buf_filename, ios::in | ios::binary);

    if(!wf)
    {
        cerr << "Could not open a file " << buf_filename << endl;
        return CONTROL_ERROR;
    }

    for(int i = 0; i < buff_size; i++)
    {
        wf.write(reinterpret_cast<char *>(&buffer[i]), sizeof(uint8_t));
    }

    wf.close();
    if(wf.bad())
    {
        cerr << "Error occured when writing to " << buf_filename << endl;
        return CONTROL_ERROR;
    }

    return CONTROL_SUCCESS;
}

Device::~Device()
{
    if(device_initialised)
    {
        device_initialised = false;
    }
}

extern "C"
Device * make_Dev(int * info, vector<string> vec)
{
    static Device dev_obj(info);
    return &dev_obj;
}
