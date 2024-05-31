// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "options.hpp"

using namespace std;

int main(int argc, char ** argv)
{
    if(argc == 1)
    {
        cout << "Use --help to get the list of options for this application." << endl
        << "Or use --list-commands to print the list of commands and their info." << endl;
        return 0;
    }

    string command_map_path = get_cmd_map_abs_path(&argc, argv);
    string device_dl_name = get_device_lib_name(&argc, argv);

    uint8_t instance_id = get_instance_id(&argc, argv); // instanceID can be present anywhere on the cmd line. Get it first

    opt_t * opt = nullptr;
    int cmd_indx = 1;
    string next_cmd = argv[cmd_indx];

    if(next_cmd[0] == '-')
    {
        opt = option_lookup(next_cmd);
        if (opt->long_name == "--help")
        {
            return print_help_menu();
        }
        else if (opt->long_name == "--version")
        {
            cout << current_host_app_version << endl;
            return 0;
        }
    }

    dl_handle_t cmd_map_handle = load_command_map_dll(command_map_path);

    if(next_cmd[0] == '-')
    {
        // This assumes that the next_cmd has not been reassigned
        // Hence opt holds the same option pointer
        if (opt->long_name == "--list-commands")
        {
            return print_command_list();
        }
    }

    string device_dl_path = get_dynamic_lib_path(device_dl_name);
    dl_handle_t device_handle = get_dynamic_lib(device_dl_path);
    int * device_init_info = get_device_init_info(cmd_map_handle, device_dl_name);

    device_fptr make_dev = get_device_fptr(device_handle);

    vector<string> device_host_args = get_device_host_arg(&argc, argv, device_dl_name); // Device related args that are passed through the command line on the host. For example, --port for control over xscope

    Device * device = make_dev(device_init_info, device_host_args);

    Command command(device, cmd_map_handle, instance_id);

    int arg_indx = cmd_indx + 1;
    next_cmd = argv[cmd_indx];

    return command.do_command(next_cmd, argv, argc, arg_indx);
}
