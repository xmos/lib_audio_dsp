// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "fileio_task.h"

static void show_usage()
{
    puts(
        "args: Specify in a single line in args.txt file.\n\n"
        "         -i     input wav file (eg. input.wav)\n\n"
        "         -o     output wav file (eg. -o output.wav)\n\n"
        "         -n     no. of output channels (eg. -o 2. If not specified, set to be same as input channels)\n\n"
        "         -h     Show this usage message and abort\n\n"
        "         -t     Input/output offset, number of samples to throw out at boot to account for pipeline length\n\n"
        );

    exit(0);
}

static void parse_one_arg(char *arg_part_1, test_config_t *test_config)
{
    char *arg_part_2 = strtok(NULL, " ");

    if(strcmp(arg_part_1, "-h") == 0)
    {
        show_usage();
    }
    else
    {
        assert(arg_part_2 != NULL);
    }

    if(strcmp(arg_part_1, "-n") == 0)
    {
        test_config->num_output_channels = atoi(arg_part_2);
    }
    else if(strcmp(arg_part_1, "-t") == 0)
    {
        test_config->num_discard_frames = atoi(arg_part_2);
    }
    else if(strcmp(arg_part_1, "-i") == 0)
    {
        test_config->input_filename = malloc(strlen(arg_part_2)+1);
        strcpy(test_config->input_filename, arg_part_2);
    }
    else if(strcmp(arg_part_1, "-o") == 0)
    {
        test_config->output_filename = malloc(strlen(arg_part_2)+1);
        strcpy(test_config->output_filename, arg_part_2);
    }
    else
    {
        printf("Error: Invalid argument %s\n", arg_part_1);
        assert(0);
    }
}


void parse_args(const char *args_file, test_config_t *test_config)
{
    FILE *fp = fopen(args_file, "r");

    // Get the length of the line first
    int count = 0;
    int ch = fgetc(fp);
    while((ch != '\n') && (ch != EOF)) {
        count += 1;
        ch = fgetc(fp);
    }
    fseek(fp, 0, SEEK_SET);
    char *args = malloc((count+1)*sizeof(char));
    // Read the full line
    assert(fgets(args, count+1, fp) != NULL);
    printf("args = %s\n", args);

    char *pch;
    pch = strtok(args, " ");
    while(pch != NULL)
    {
        parse_one_arg(pch, test_config);
        pch = strtok(NULL, " ");
    }
}
