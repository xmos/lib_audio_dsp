#ifndef CMDS_H
#define CMDS_H

%for name, data in cmd_map.items():
// ${name} module commands
<% cmd_id = 1 %>\
%for field_name, field_data in data.items():
#define CMD_${name.upper()}_${field_name.upper()}             ${cmd_id}
<% cmd_id = cmd_id + 1 %>\
%endfor
#define NUM_CMDS_${name.upper()}    ${len(data)}    // Number of commands in the ${name} module

%endfor

#endif
