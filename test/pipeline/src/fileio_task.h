// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#define FILEREAD_CHUNK_SIZE (1024)
typedef struct
{
    char *input_filename;
    char *output_filename;
    int num_output_channels;
}test_config_t;

void parse_args(const char *args_file, test_config_t *test_config);

