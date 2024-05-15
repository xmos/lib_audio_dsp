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

    FILE *in = _fopen("../rv_sig_48k.bin", "rb");
    FILE *out = _fopen("rv_sig_out.bin", "wb");
    FILE *info = _fopen("rv_info.bin", "rb");

    fseek(in, 0, SEEK_END);
    int in_len = ftell(in) / sizeof(int32_t);
    fseek(in, 0, SEEK_SET);

    int32_t pregain, wet, dry, feedback, damping;

    fread(&pregain, sizeof(int32_t), 1, info);
    fread(&wet, sizeof(int32_t), 1, info);
    fread(&dry, sizeof(int32_t), 1, info);
    fread(&feedback, sizeof(int32_t), 1, info);
    fread(&damping, sizeof(int32_t), 1, info);
    fclose(info);

    uint8_t reverb_heap[ADSP_RVR_HEAP_SZ(FS, MAX_ROOM)] = {0};
    reverb_room_t rv;
    rv.pre_gain = pregain;
    rv.wet_gain = wet;
    rv.dry_gain = dry;

    adsp_reverb_room_init_filters(&rv, fs, max_room_size, feedback, damping, reverb_heap);
    adsp_reverb_room_set_room_size(&rv, room_size);

    for (int i = 0; i < in_len; i++)
    {
        int32_t samp = 0, samp_out = 0;
        fread(&samp, sizeof(int32_t), 1, in);
        samp_out = adsp_reverb_room(&rv, samp);
        fwrite(&samp_out, sizeof(int32_t), 1, out);
    }

    fclose(in);
    fclose(out);

    return 0;
}
