// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "utils.hpp"
#include <iomanip>

using namespace std;

#define PI_VALUE  3.14159265358979323846f

string command_param_type_name(const cmd_param_type_t type)
{
    string tstr;

    switch (type)
    {

    case TYPE_UINT8:
        tstr = "uint8";
        break;

    case TYPE_INT32:
        tstr = "int32";
        break;

    case TYPE_UINT32:
        tstr = "uint32";
        break;

    case TYPE_FLOAT:
        tstr = "float";
        break;

    default:
        cerr << "Unsupported parameter type" << endl;
        exit(HOST_APP_ERROR);
    }

    return tstr;
}

cmd_param_t cmd_arg_str_to_val(const cmd_param_type_t type, const char * str)
{
    cmd_param_t val;
    try{
        switch(type)
        {
        case TYPE_UINT8:
        {
            int32_t tmp = stoi(str, nullptr, 0);
            if ((tmp > UINT8_MAX) || (tmp < 0))
            {
                throw out_of_range("");
            }
            val.ui8 = static_cast<uint8_t>(tmp);
            break;
        }
        case TYPE_INT32:
            val.i32 = stoi(str, nullptr, 0);
            break;

        case TYPE_UINT32:
            val.ui32 = stoul(str, nullptr, 0);
            break;

        case TYPE_FLOAT:
            val.f = stof(str);
            break;

        default:
            cerr << "Unsupported parameter type" << endl;
            exit(HOST_APP_ERROR);
        }
    }
    catch(const out_of_range & ex)
    {
        static_cast<void>(ex);
        cerr << "Value " << str << " is out of range of " << command_param_type_name(type) << " type"<< endl;
        exit(HOST_APP_ERROR);
    }
    catch(const invalid_argument & ex)
    {
        static_cast<void>(ex);
        cerr << "Argument " << str << " is invalid" << endl;
        exit(HOST_APP_ERROR);
    }
    return val;
}

size_t get_num_bytes_from_type(const cmd_param_type_t type)
{
    size_t num_bytes;
    switch(type)
    {
    case TYPE_UINT8:
        num_bytes = 1;
        break;
    case TYPE_INT32:
    case TYPE_UINT32:
    case TYPE_FLOAT:
        num_bytes = 4;
        break;
    default:
        cerr << "Unsupported parameter type" << endl;
        exit(HOST_APP_ERROR);
    }
    return num_bytes;
}
