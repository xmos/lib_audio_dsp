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
    size_str = "[" + field_data['size'] + "]" if "size" in field_data else ""
%>\
    ${field_data["type"]} ${attrib_str}${field_name}${size_str};
%endfor
}${name}_config_t;

#endif

