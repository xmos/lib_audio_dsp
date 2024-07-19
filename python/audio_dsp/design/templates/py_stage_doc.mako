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
The following control parameters are available for the ${module}.${cl} Stage:

## .. list-table::

* - index
  - cmd_id
  - payload_len
  - description

% for field_name, field_data in class_data[cl].items():
<%
    attrib_str = f'{field_data["attribute"]} ' if "attribute" in field_data else ""
    size_str = "[" + str(field_data['size']) + "]" if "size" in field_data else ""
    help_str = f'{field_data["help"]} ' if "help" in field_data else ""
%>\
${help_str}
${field_data["type"]} ${attrib_str}${field_name}${size_str};

% endfor  ## field_name, field_data in class_data[cl].items()

% else:  ## class_data[cl]

The ${module}.${cl} Stage has no runtime controllable parameters.

% endif ## class_data[cl]

% endfor ## cl in classes
