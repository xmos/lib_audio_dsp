// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "cmd_num_values.h"
#include "cmds.h"

static cmd_t commands[] =
{
%for name, data in cmd_map.items():
    %for field_name, field_data in data.items():
<%
    field_data["size"] = field_data["size"] if "size" in field_data else 1
    cmd_type = {'int32_t': 'TYPE_INT32', 'int16_t': 'TYPE_INT16', 'int8_t': 'TYPE_CHAR', 'uint8_t': 'TYPE_UINT8', 'float': 'TYPE_FLOAT', 'int': 'TYPE_INT32', 'uint32_t': 'TYPE_UINT32', 'float_s32_t': 'TYPE_INT32', 'int32_t*': 'TYPE_INT32'}
%>\
 {0xff, "${name.upper()}_${field_name.upper()}", ${cmd_type[field_data["type"]]}, CMD_${name.upper()}_${field_name.upper()}, CMD_READ_WRITE, NUM_VALUES_${name.upper()}_${field_name.upper()}, "${field_data["help"].strip()}", false},
    %endfor

%endfor
};
