// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "dsp/adsp.h"

#define PRINT_INIT 0
#define FS 48000
#define MAX_ROOM 1.0

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
    float const fs = FS;
    float const max_room_size = MAX_ROOM;
    float const room_size = 1.0;
    float const decay = 1.0;
    float const damping = 1.0;
    float const wet_gain_db = -1.0;
    float const dry_gain_db = -1.0;
    float const pregain = 0.015;

    int32_t wet_gain = adsp_reverb_calc_wet_gain(wet_gain_db, pregain);
    int32_t dry_gain = adsp_reverb_calc_dry_gain(dry_gain_db);

    FILE *in = _fopen("../rv_sig_48k.bin", "rb");
    FILE *out = _fopen("rv_sig_out.bin", "wb");

    fseek(in, 0, SEEK_END);
    int in_len = ftell(in) / sizeof(int32_t);
    fseek(in, 0, SEEK_SET);

    uint8_t reverb_heap[ADSP_RV_HEAP_SZ(FS, MAX_ROOM)] = {0};
    reverb_room_t reverb = adsp_reverb_room_init(fs,
                                                 max_room_size, room_size,
                                                 decay, damping, wet_gain,
                                                 dry_gain, pregain,
                                                 reverb_heap);

    for (int i = 0; i < in_len; i++)
    {
        int32_t samp = 0, samp_out = 0;
        fread(&samp, sizeof(int32_t), 1, in);
        samp_out = adsp_reverb_room(&reverb, samp);
        fwrite(&samp_out, sizeof(int32_t), 1, out);
    }

    fclose(in);
    fclose(out);

    return 0;
}
