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

.. list-table::
  :widths: 54, 21, 25

  * - Command ID macro
    - Payload length
    - Description

<% cmd_id = 1 %>\
% for field_name, field_data in class_data[cl].items():
<%
    import re
    size_str = "*[" + str(field_data['size']) + "]" if "size" in field_data else ""
    help_str = f'{field_data["help"]} ' if "help" in field_data else ""
    safe_name = cl.replace("RMS", "Rms")
    snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', safe_name).upper()
%>\
  ## * - ${cmd_id}
  * - CMD_${snake_name}_${field_name.upper()}
    - ``sizeof(${field_data["type"]})${size_str}``
    - ${help_str}

<% cmd_id = cmd_id + 1 %>\
% endfor  ## field_name, field_data in class_data[cl].items()

% else:  ## class_data[cl]

The ${cl} Stage has no runtime controllable parameters.

% endif ## class_data[cl]

% endfor ## cl in classes


