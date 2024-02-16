// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef ${name.upper()}_CONFIG_H
#define ${name.upper()}_CONFIG_H

%for i in includes:
#include <${i}>
%endfor

%for d in defines:
#define ${d}    ${defines[d]}
%endfor

typedef struct
{
%for field_name, field_data in data.items():
<%
    attrib_str = f'{field_data["attribute"]} ' if "attribute" in field_data else ""
    size_str = "[" + str(field_data['size']) + "]" if "size" in field_data else ""
%>\
    ${field_data["type"]} ${attrib_str}${field_name}${size_str};
%endfor
}${name}_config_t;

#endif

