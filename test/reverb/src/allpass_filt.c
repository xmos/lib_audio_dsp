// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "dsp/adsp.h"

FILE *_fopen(char *fname, char *mode)
{
    FILE *fp = fopen(fname, mode);
    if (fp == NULL)
    {
        printf("Error opening a file\n");
        exit(1);
    }
    return fp;
}

int main()
{
    uint32_t const max_delay = 605;
    uint32_t const starting_delay = 605;
    int32_t const feedback_gain = Q31(0.5);
    uint8_t heap[605 * sizeof(int32_t)] = {0};
    mem_manager_t mem_man_mock = {heap, 605 * sizeof(int32_t), 0};

    FILE *in = _fopen("../rv_sig_48k.bin", "rb");
    FILE *out = _fopen("rv_sig_out.bin", "wb");

    fseek(in, 0, SEEK_END);
    int in_len = ftell(in) / sizeof(int32_t);
    fseek(in, 0, SEEK_SET);

    allpass_fv_t ap = allpass_fv_init(max_delay, starting_delay, feedback_gain,
                                      &mem_man_mock);

    for (int i = 0; i < in_len; i++)
    {
        int32_t samp = 0, samp_out = 0;
        fread(&samp, sizeof(int32_t), 1, in);
        samp_out = allpass_fv(&ap, samp);
        fwrite(&samp_out, sizeof(int32_t), 1, out);
    }

    fclose(in);
    fclose(out);

    return 0;
}
