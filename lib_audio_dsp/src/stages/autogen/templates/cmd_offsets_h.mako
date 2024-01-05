#ifndef CMD_OFFSETS_H
#define CMD_OFFSETS_H

#include "adsp_module.h"

#include "cmds.h"
%for name in cmd_map:
#include "${name}_config.h"
%endfor

%for name, data in cmd_map.items():
// Offset and size of fields in the ${name}_config_t structure
static module_config_offsets_t ${name}_config_offsets[] = {
%for field_name, field_data in data.items():
<% field_data["size"] = field_data["size"] if "size" in field_data else 1 %>\
{.cmd_id=CMD_${name.upper()}_${field_name.upper()}, .offset=offsetof(${name}_config_t, ${field_name}), .size=sizeof(${field_data["type"]}) * ${field_data["size"]}},
%endfor
};
%endfor

static module_config_offsets_t *ptr_module_offsets[] = {
%for name, data in cmd_map.items():
${name}_config_offsets,
%endfor
};

#endif
