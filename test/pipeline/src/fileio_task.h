#ifndef FILEIO_TASK_H
#define FILEIO_TASK_H

#define FILEREAD_CHUNK_SIZE (1024)
typedef struct
{
    char *input_filename;
    char *output_filename;
    int num_output_channels;
}test_config_t;

void parse_args(const char *args_file, test_config_t *test_config);

#endif
