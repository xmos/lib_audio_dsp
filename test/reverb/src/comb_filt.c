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
    uint32_t const max_delay = 1760;
    uint32_t const starting_delay = 1760;
    int32_t const feedback_gain = Q31(0.98);
    int32_t const damping = Q31(1.0);
    uint8_t heap[1760 * sizeof(int32_t)] = {0};
    mem_manager_t mem_man_mock = {heap, 1760 * sizeof(int32_t), 0};

    FILE *in = _fopen("../rv_sig_48k.bin", "rb");
    FILE *out = _fopen("rv_sig_out.bin", "wb");

    fseek(in, 0, SEEK_END);
    int in_len = ftell(in) / sizeof(int32_t);
    fseek(in, 0, SEEK_SET);

    comb_fv_t comb = comb_fv_init(max_delay, starting_delay, feedback_gain,
                                  damping, &mem_man_mock);

    for (int i = 0; i < in_len; i++)
    {
        int32_t samp = 0, samp_out = 0;
        fread(&samp, sizeof(int32_t), 1, in);
        samp_out = comb_fv(&comb, samp);
        fwrite(&samp_out, sizeof(int32_t), 1, out);
    }

    fclose(in);
    fclose(out);

    return 0;
}
