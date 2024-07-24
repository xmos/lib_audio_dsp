${"#"*len(title)}
${title}
${"#"*len(title)}

${docstring}

% for cl in classes:
% if len(classes) != 1:
${"="*len(cl)}
${cl}
${"="*len(cl)}
% endif

.. autoclass:: ${module}.${cl}
    :noindex:
    :members:

.. rubric:: Control

% if class_data[cl]:
The following runtime control parameters are available for the ${cl} Stage:
<%
  row_list = []
  max_cmd = len("Command ID macro")
  max_pay = len("Payload length")
  max_help = len("Description")
  for field_name, field_data in class_data[cl].items():
    import re
    size_str = "*[" + str(field_data['size']) + "]" if "size" in field_data else ""
    help_str = f'{field_data["help"].strip()}' if "help" in field_data else ""
    safe_name = cl.replace("RMS", "Rms")
    snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', safe_name).upper()
    cmd_str = f"CMD_{snake_name}_{field_name.upper()}"
    payload_str = f"``sizeof({field_data['type']}){size_str}``"
    this_row = [cmd_str, payload_str, help_str]
    row_list.append(this_row)
  
    if len(cmd_str) > max_cmd:
      max_cmd = len(cmd_str)
    if len(payload_str) > max_pay:
      max_pay = len(payload_str)
    if len(help_str) > max_help:
      max_help = len(help_str)
  
  page_width = 80
  cmd_width = int((max_cmd/page_width)*100)
  pay_width = int((max_pay/page_width)*100)
  help_width = 100 - cmd_width - pay_width

%>
##  do the printing, use ljust to pad to max size
.. table::
  :widths: ${cmd_width}, ${pay_width}, ${help_width} 
  
  ${"="*max_cmd}  ${"="*max_pay}  ${"="*max_help}
  ${"Command ID macro".ljust(max_cmd)}  ${"Payload length".ljust(max_pay)}  ${"Description".ljust(max_help)}
  ${"="*max_cmd}  ${"="*max_pay}  ${"="*max_help}
% for row in row_list:
  ${row[0].ljust(max_cmd)}  ${row[1].ljust(max_pay)}  ${row[2].ljust(max_help)}
% endfor  ## row in row_list
  ${"="*max_cmd}  ${"="*max_pay}  ${"="*max_help}

% else:  ## class_data[cl]

The ${cl} Stage has no runtime controllable parameters.

% endif ## class_data[cl]

% endfor ## cl in classes


