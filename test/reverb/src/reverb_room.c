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
    uint32_t const n_chans = 1;
    float const fs = 48000;
    float const max_room_size = 1.0;
    float const room_size = 1.0;
    float const decay = 1.0;
    float const damping = 1.0;
    float const wet_gain_db = -1.0;
    float const dry_gain_db = -1.0;
    float const pregain = 0.015;

    uint8_t reverb_heap[RV_HEAP_SZ(48000, 1.0f)] = {0};
    reverb_room_t reverb = adsp_reverb_room_init(n_chans, fs,
                                                 max_room_size, room_size,
                                                 decay, damping, wet_gain_db,
                                                 dry_gain_db, pregain,
                                                 reverb_heap);

    for (int i = 0; i < 10; i++)
    {
        adsp_reverb_room(&reverb, 0xFFFFFFFF, 0);
    }

    return 0;
}
