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

/** 
 * This structure allows for real time control of the ${name} stage.
 * It should be passed to the ${name}_control function.
 */
typedef struct
{
%for field_name, field_data in data.items():
<%
    attrib_str = f'{field_data["attribute"]} ' if "attribute" in field_data else ""
    size_str = "[" + str(field_data['size']) + "]" if "size" in field_data else ""
    help_str = f'{field_data["help"]} ' if "help" in field_data else ""
%>\
<%block filter="wrap_helpstr">
    ${help_str}
</%block>
    ${field_data["type"]} ${attrib_str}${field_name}${size_str};

%endfor
}${name}_config_t;

#endif

<%!
# This is a function to wrap long help descriptions
import textwrap
def wrap_helpstr(text):
    line_len = 80
    max_len = line_len - 4 - 4 - 3
    if len(text) > line_len:
        return "    /**\n     * " + '\n'.join(textwrap.wrap(text.strip(), line_len, subsequent_indent='     * ')).strip() + "\n     */"
    else:
        return f"    /** {text.strip()} */"
%>