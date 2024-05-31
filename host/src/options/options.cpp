// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "options.hpp"
#include <fstream>
#include <iomanip>
#include <ctype.h>

#if (defined(__APPLE__) || defined(_WIN32))
#include <sstream>
#endif

using namespace std;

opt_t options[] = {
    {"--help",                    "-h",        "Display this information"                                                                                                                       },
    {"--version",                 "-v",        "Print the current version of this application",                                                                                                 },
    {"--list-commands",           "-l",        "Print list of the available commands"                                                                                                           },
    {"--use",                     "-u",        "Use specific hardware protocol: xSCOPE and USB are available to use. Check documentation  for the supported protocol of each platform"},
    {"--command-map-path",        "-cmp",      "Use specific command map path, the path is relative to the working dir"                                                                         },
    {"--instance-id",             "-i",        "Module instance ID that the control command is directed to"                                                                                     },
    {"--port",                    "-p",        "Port number on which to connect to the device when doing control over xscope"}
};
size_t num_options = end(options) - begin(options);

extern size_t num_commands;

string get_cmd_map_abs_path(int * argc, char ** argv)
{
    string cmd_map_rel_path = default_command_map_name;
    string cmd_map_abs_path = "";
    opt_t * cmp_opt = option_lookup("--command-map-path");
    size_t index = argv_option_lookup(*argc, argv, cmp_opt);
    if(index != 0)
    {
        // Use path given via CLI
        cmd_map_rel_path = argv[index + 1];
        remove_opt(argc, argv, index, 2);
        cmd_map_abs_path = convert_to_abs_path(cmd_map_rel_path);
    }
    else
    {
        cmd_map_abs_path = get_dynamic_lib_path(cmd_map_rel_path);
    }
    return cmd_map_abs_path;
}

opt_t * option_lookup(const string str)
{
    string low_str = to_lower(str);
    for(size_t i = 0; i < num_options; i++)
    {
        opt_t * opt = &options[i];
        if ((low_str == opt->long_name) || (low_str == opt->short_name))
        {
            return opt;
        }
    }

    cerr << "Option " << str << " does not exist." << endl;

    exit(HOST_APP_ERROR);
    return nullptr;
}

string get_device_lib_name(int * argc, char ** argv)
{
    string lib_name = default_driver_name;
    opt_t * use_opt = option_lookup("--use");
    size_t index = argv_option_lookup(*argc, argv, use_opt);
    if(index == 0)
    {
        // could not find --use, using default driver name
        return lib_name;
    }
    else
    {
        string protocol_name = argv[index + 1];
        if (to_upper(protocol_name) == "USB")
        {
            lib_name = device_usb_dl_name;
        }
        else if (to_upper(protocol_name) == "XSCOPE")
        {
            lib_name = device_xscope_dl_name;
        }
        else
        {
            cout << "Could not find " << to_upper(protocol_name) << " in supported protocols"
            << endl << "Will use XSCOPE by default" << endl;
        }
        remove_opt(argc, argv, index, 2);
        return lib_name;
    }
}

vector<string> get_device_host_arg(int * argc, char ** argv, string lib_name)
{
    if(lib_name == device_xscope_dl_name)
    {
        opt_t * use_opt = option_lookup("--port");
        size_t index_port = argv_option_lookup(*argc, argv, use_opt);
        if(index_port == 0)
        {
            cerr << "No port specified when doing control over xscope. Provide the port number using the --port option." << endl;
            exit(HOST_APP_ERROR);
        }
        else
        {
            string port_num = argv[index_port + 1];
            remove_opt(argc, argv, index_port, 2);
            vector<string> v = {port_num};
            return v;
        }
    }
    else
    {
        vector<string> v;
        return v;
    }
}

uint8_t get_instance_id(int * argc, char ** argv)
{
    opt_t *band_opt = option_lookup("--instance-id");
    size_t index = argv_option_lookup(*argc, argv, band_opt);
    if (index == 0) // --instance-id not provided. Use the one in command map.
    {
        return INVALID_INSTANCE_ID;
    }
    else
    {
        if (isdigit(argv[index+1][0])) // Get the actual index that follows the --instance-id option
        {
            int instance = atoi(argv[index+1]);
            remove_opt(argc, argv, index, 2);
            return instance;
        }
        else
        {
            cerr << "No instance ID provided after the --instance-id option." << endl;
            exit(HOST_APP_ERROR);
        }
    }
}

control_ret_t print_help_menu()
{
    size_t longest_short_opt = 0;
    size_t longest_long_opt = 0;
    for(opt_t opt : options)
    {
        size_t short_len = opt.short_name.length();
        size_t long_len = opt.long_name.length();
        longest_short_opt = (short_len > longest_short_opt) ? short_len : longest_short_opt;
        longest_long_opt = (long_len > longest_long_opt) ? long_len : longest_long_opt;
    }
    size_t long_opt_offset = longest_short_opt + 5;
    size_t info_offset = long_opt_offset + longest_long_opt + 4;
    // Getting current terminal width here to set the cout line limit
    const size_t hard_stop = get_term_width();

    // Please avoid lines which have more than 80 characters
    cout << "usage: dsp_host [ command | option ]" << endl
    << setw(78) << "[ -u <protocol> ] [ -cmp <path> ] [ -br ] [ command | option ]" << endl
    << endl << "Current application version is " << current_host_app_version << "."
    << endl << "You can use --use or -u option to specify protocol you want to use"
    << endl << "or call the option/command directly using default control protocol."
    << endl << "Default control protocol is USB."
    << endl << "You can use --command-map-path or -cmp to specify the command_map object to use."
    << endl << endl << "Options:" << endl;
    for(opt_t opt : options)
    {
        size_t short_len = opt.short_name.length();
        size_t long_len = opt.long_name.length();
        size_t info_len = opt.info.length();
        size_t first_word_len = opt.info.find_first_of(' ');
        int first_space = long_opt_offset - short_len + long_len;
        int second_space = info_offset - long_len - long_opt_offset + first_word_len;
        int num_spaces = 2; // adding two black spaces at the beggining to make it look nicer

        cout << setw(num_spaces) << " " << opt.short_name << setw(first_space)
        << opt.long_name << setw(second_space);

        stringstream ss(opt.info);
        string word;
        size_t curr_pos = info_offset + num_spaces;
        while(ss >> word)
        {
            size_t word_len = word.length();
            size_t future_pos = curr_pos + word_len + 1;
            if(future_pos >= hard_stop)
            {
                cout << endl << setw(info_offset + word_len + num_spaces) << word << " ";
                curr_pos = info_offset + word_len + num_spaces + 1;
            }
            else
            {
                cout << word << " ";
                curr_pos = future_pos;
            }
        }
        cout << endl << endl;
    }
    return CONTROL_SUCCESS;
}

control_ret_t print_command_list()
{
    size_t longest_command = 0;
    size_t longest_rw = 10; // READ/WRITE
    size_t longest_args = 2; // double digits
    size_t longest_type = 6; // uint32
    size_t longest_info = 0;
    for(size_t i = 0; i < num_commands; i ++)
    {
        cmd_t cmd = {0};
        init_cmd(&cmd, "_", i);
        // skipping hidden commands
        if(cmd.hidden_cmd)
        {
            continue;
        }
        size_t name_len = cmd.cmd_name.length();
        size_t info_len = cmd.info.length();
        longest_command = (longest_command < name_len) ? name_len : longest_command;
        longest_info = (longest_info < info_len) ? info_len : longest_info;
    }
    size_t rw_offset = longest_command + 2;
    size_t args_offset = rw_offset + longest_rw + 2;
    size_t type_offset = args_offset + longest_args + 2;
    size_t info_offset = type_offset + longest_type + 2;
    // Getting current terminal width here to set the cout line limit
    const size_t hard_stop = get_term_width();

    for(size_t i = 0; i < num_commands; i ++)
    {
        cmd_t cmd = {0};
        init_cmd(&cmd, "_", i);
        // skipping hidden commands
        if(cmd.hidden_cmd)
        {
            continue;
        }
        // name   rw   args   type   info
        size_t name_len = cmd.cmd_name.length();
        string rw = command_rw_type_name(cmd.rw);
        size_t rw_len = rw.length();
        size_t args_len = to_string(cmd.num_values).length();
        string type = command_param_type_name(cmd.type);
        size_t type_len = type.length();
        size_t first_word_len = cmd.info.find_first_of(' ');

        int first_space = rw_offset - name_len + rw_len;
        int second_space = args_offset - rw_len - rw_offset + args_len;
        int third_space = type_offset - args_len - args_offset + type_len;
        int fourth_space = info_offset - type_len - type_offset + first_word_len;

        cout << cmd.cmd_name << setw(first_space) << rw
        << setw(second_space) << cmd.num_values << setw(third_space)
        << type << setw(fourth_space);

        stringstream ss(cmd.info);
        string word;
        size_t curr_pos = info_offset;
        while(ss >> word)
        {
            size_t word_len = word.length();
            size_t future_pos = curr_pos + word_len + 1;
            if(future_pos >= hard_stop)
            {
                cout << endl << setw(info_offset + word_len) << word << " ";
                curr_pos = info_offset + word_len;
            }
            else
            {
                cout << word << " ";
                curr_pos = future_pos;
            }
        }
        cout << endl << endl;
    }
    return CONTROL_SUCCESS;
}
