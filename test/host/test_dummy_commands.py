# Copyright 2023 XMOS LIMITED.
# This Software is subject to the terms of the XCORE VocalFusion Licence.

import test_utils

num_vals = 20
num_frames = 10
float_cmd = "CMD_FLOAT"
int32_cmd = "CMD_INT32"
uint32_cmd = "CMD_UINT32"
rads_cmd = "CMD_RADS"
uint8_cmd = "CMD_UINT8"
char_cmd = "CMD_CHAR"
small_cmd = "CMD_SMALL"

def single_command_test(host_bin, control_protocol, cwd, cmd_name, cmd_vals):
    # Set and get values within the command range and test closeness
    out_list = test_utils.execute_command(host_bin, control_protocol, cwd, cmd_name, cmd_vals=cmd_vals)

    for i in range(len(cmd_vals)):
        read_output = float(out_list[i])
        # ~20 bits should be good for this test
        rtol = 1e-6
        e = 1e-30
        abs_diff = abs(cmd_vals[i] - read_output)
        rel_error = abs(abs_diff/(cmd_vals[i] + e))
        assert rel_error < rtol

def test_dummy_commands():
    test_dir, host_bin, control_protocol, _ = test_utils.get_dummy_files()
    print("\n")

    with open(test_dir / 'test_buf.bin', 'w'):
        pass
    
    for i in range(num_frames):
        vals = test_utils.gen_rand_array('float', -2147483648, 2147483647)
        single_command_test(host_bin, control_protocol, test_dir, float_cmd, vals)

        vals = test_utils.gen_rand_array('int', -2147483648, 2147483647)
        single_command_test(host_bin, control_protocol, test_dir, int32_cmd, vals)

        vals = test_utils.gen_rand_array('int', 0, 4294967295)
        single_command_test(host_bin, control_protocol, test_dir, uint32_cmd, vals)

        vals = test_utils.gen_rand_array('int', 0, 255)
        single_command_test(host_bin, control_protocol, test_dir, uint8_cmd, vals)

        output = test_utils.execute_command(host_bin, control_protocol, test_dir, char_cmd)
        sentence = " ".join(str(word) for word in output)

        assert sentence == "Hello New World!\0\0\0\0"
