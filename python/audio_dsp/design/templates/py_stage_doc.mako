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

${cl} Control
${"="*len(cl)}========

% if class_data[cl]:
The following runtime command ids are available for the ${cl} Stage. For
details on reading and writing these commands, see the Run-Time Control User Guide.

<%
  row_list = []
  max_cmd = len("Command ID macro") + 2 # add 2 so the title doesn't ever get split in the PDF
  max_pay = len("Payload length")
  max_help = len("Description")
  for field_name, field_data in class_data[cl].items():
    import re
    size_str = "*[" + str(field_data['size']) + "]" if "size" in field_data else ""
    help_str = f'{field_data["help"].strip()}' if "help" in field_data else ""
    if "rw_type" in field_data and field_data["rw_type"] == "CMD_READ_ONLY":
      help_str += " This command is read only. When sending a write control command, it will be ignored."
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
  
  page_width = 74 # this is a guesstimate of number of chars in a page
  cmd_width = int((max_cmd/page_width)*100)
  pay_width = int(((max_pay - 5)/page_width)*100) # subtract ` chars and compensate for font (ish)
  help_width = 100 - pay_width

%>
##  do the printing, use ljust to pad to max size
.. table::
  :widths: ${help_width}, ${pay_width} 

  ${"="*max_help}  ${"="*max_pay}
  ${"Control parameter".ljust(max_help)}  ${"Payload length".ljust(max_pay)}
  ${"="*max_help}  ${"="*max_pay}
% for row in row_list:
  ${row[0].ljust(max_help)}  ${row[1].ljust(max_pay)}
% if "²" in row[2]:  ## ljust in Mako ignores ², no idea why
  ${row[2].ljust(max_help + 1)}  ${'\\'.ljust(max_pay)}
% else:
  ${row[2].ljust(max_help)}  ${'\\'.ljust(max_pay)}
%endif  ## "²" in row[2]:
% if row_list.index(row) < len(row_list) - 1:
  ## don't print a blank row at the end
  |
  ${"-"*max_help}--${"-"*max_pay}
% endif  ## row_list.index(row) < len(row_list) - 1
% endfor  ## row in row_list
  ${"="*max_help}  ${"="*max_pay}

% else:  ## class_data[cl]

The ${cl} Stage has no runtime controllable parameters.

% endif ## class_data[cl]

% endfor ## cl in classes


