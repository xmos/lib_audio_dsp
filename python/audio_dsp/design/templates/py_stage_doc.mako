.. _${title}_stages:

${"#"*(len(title) + 7)}
${title} Stages
${"#"*(len(title) + 7)}

${docstring}

% for cl in classes:
.. _${cl}_stage:

${"="*len(cl)}
${cl}
${"="*len(cl)}

.. autoclass:: audio_dsp.stages.${cl}
    :noindex:
    :members:
    :inherited-members: Stage
<%

    ## Get the parameter type from the stage.set_parameters method
    import importlib
    import inspect
    from typing import get_type_hints
    import types
    import audio_dsp
    # Import the stage class
    stages_mod = importlib.import_module('audio_dsp.stages')
    stage_cls = getattr(stages_mod, cl)
    # Get the set_parameters method
    set_params = getattr(stage_cls, 'set_parameters')
    # Get the type hints for the method
    hints = get_type_hints(set_params)
    # Assume the first argument after 'self' is the model class
    params = inspect.signature(set_params).parameters
    param_names = list(params.keys())
    # Skip 'self', get the next parameter
    param_name = param_names[1]
    model_cls = hints.get(param_name)

    if model_cls == audio_dsp.design.stage.StageParameterType:
        # generic StageParameterType, so no specific model class
        model_cls = []
    elif type(model_cls) == types.UnionType:
        # If it's a Union, there's multiple types
        model_cls = list(model_cls.__args__)
    else:
        model_cls = [model_cls]

    for i, cls in enumerate(model_cls):
        # get the full class path
        model_cls[i] = f"{cls.__module__}.{cls.__name__}"

%>
% for param_cls in model_cls:
.. autopydantic_model:: ${param_cls}
    :noindex:
    :members:
    :model-show-config-summary: False
    :model-show-field-summary: False
% endfor ## for cls in model_cls


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
% if len(row_list) > 6:
  :class: longtable
%endif

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


