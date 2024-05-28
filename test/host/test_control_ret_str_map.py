# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

## Autogenerate the control_ret_t string mapping from the device_control_shared.h file and check that
## it's the same as what's committed in src/utils/control_ret_str_map.h

import parse_control_ret_t
from pathlib import Path
import subprocess

def test_control_ret_str_map():
    src_dir = Path(__file__).parents[2] / "host/src"
    device_control_shared_h = Path(src_dir) / "device_control/api/device_control_shared.h"
    assert device_control_shared_h.is_file(), f"Error: {device_control_shared_h} is not a valid file"
    control_ret_dict, control_ret_str_map = parse_control_ret_t.parse_control_ret_t(device_control_shared_h)
    test_file = "src.autogen/test.h"
    parse_control_ret_t.create_enum_str_map_h_file(control_ret_str_map, test_file)
    diff_command = f"diff -w {test_file} {Path(src_dir)}/utils/control_ret_str_map.h"
    subprocess.run(diff_command.split(), check=True)

if __name__ == "__main__":
    test_control_ret_str_map()
