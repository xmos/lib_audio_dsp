# Copyright 2023 XMOS LIMITED.
# This Software is subject to the terms of the XCORE VocalFusion Licence.
import re
import sys
from pathlib import Path
import os

def parse_control_ret_t(file_dev_ctrl_h):
    with open(file_dev_ctrl_h) as f:
        struct_lines = ""
        lines = f.readlines()
        save_struct = False
        struct_found = False
        for line in lines:
            if re.match(r"^\s*typedef enum\s*\{", line):
                struct_lines = ""
                save_struct = True
            if save_struct and re.match(r"\s*\}\s*control_ret_t;", line):
                struct_lines += line
                struct_lines += "\n"
                struct_found = True
                save_struct = False
                break

            if save_struct:
                struct_lines += line
        if struct_found is False or save_struct:
            print(f"Error: control_ret_t enum in {file_dev_ctrl_h} is not formatted correctly")
            assert False
    
    ## Step 1: Parse into a {'enum_str':value} dictionary
    prev = -1
    enum_dict = {}
    for line in struct_lines.split('\n'):
        if '{' not in line and '}' not in line:
            m = re.match(r"^\s*([^\s,]+)\s*=?\s*([^\s,]+)?", line)
            if m:
                err_str = str(m.group(1))
                val = prev + 1
                if m.group(2):
                    val = int(m.group(2))
                prev = val
                enum_dict[err_str] = val

    ## Step 2: From the enum dictionary, also generate the enum to string map
    # This is an array of strings where the array is indexed by the enum value, such that array[enum_value] = "enum_string_at_that_value"
    # Sort based on value
    sorted_by_val_list = sorted(enum_dict.items(), key=lambda item:item[1])
    list_size = sorted_by_val_list[-1][1] + 1
    assert list_size <= 256, f"Error: Cannot have enum values above 256 in control_ret_t enum. Current highest enum value = {list_size}"
    # Create a list of empty strings
    enum_to_string_map = ["UNUSED_ENUM"] * 256

    for i in sorted_by_val_list:
        enum_to_string_map[i[1]] = i[0]

    return enum_dict, enum_to_string_map

def create_enum_str_map_h_file(enum_to_string_map, filename="test_control_ret_map.h"):
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    pkg_dir = Path(__file__).parent
    # Load template
    env = Environment(
        loader=FileSystemLoader(f"{pkg_dir}/templates"),
        autoescape=select_autoescape(),
        trim_blocks=True
    )
    print(Path(filename).parent)
    os.makedirs(Path(filename).parent, exist_ok=True)
    with open(filename, 'w') as fh:
            template = env.get_template("control_ret_str_map.h")
            fh.write(template.render(str_map=enum_to_string_map))

if __name__ == "__main__":
    control_ret_dict, control_ret_str_map = parse_control_ret_t(sys.argv[1])
    create_enum_str_map_h_file(control_ret_str_map, "src.autogen/control_ret_str_map.h")
