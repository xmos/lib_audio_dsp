// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

typedef struct
{
    char *input_filename;
    char *output_filename;
    int num_discard_frames;
    int num_output_channels;
}test_config_t;

void parse_args(const char *args_file, test_config_t *test_config);

