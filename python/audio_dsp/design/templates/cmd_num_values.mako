// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#ifndef CMD_NUM_VALUES_H
#define CMD_NUM_VALUES_H

%for name, data in cmd_map.items():
    %for field_name, field_data in data.items():
<%
    if "size" in field_data:
        size = field_data["size"]
    elif field_data["type"] == "float_s32_t":
        size = 2
    else:
        size = 1
%>\
#define NUM_VALUES_${name.upper()}_${field_name.upper()} ${size}

    %endfor
%endfor
#endif
