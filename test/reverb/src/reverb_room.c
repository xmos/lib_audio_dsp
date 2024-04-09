// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "dsp/adsp.h"

#define FS 48000.0
#define MAX_ROOM_SIZE 1.0
#define PRINT_INIT 1

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
    float const fs = FS;
    float const max_room_size = MAX_ROOM_SIZE;
    float const room_size = 1.0;
    float const decay = 1.0;
    float const damping = 1.0;
    float const wet_gain_db = -1.0;
    float const dry_gain_db = -1.0;
    float const pregain = 0.015;
    uint32_t const channel = 0;

    int32_t wet_gain = adsp_reverb_calc_wet_gain(wet_gain_db, pregain);
    int32_t dry_gain = adsp_reverb_calc_dry_gain(dry_gain_db);

    FILE *in = _fopen("../rv_sig_48k.bin", "rb");
    FILE *out = _fopen("rv_sig_out.bin", "wb");

    fseek(in, 0, SEEK_END);
    int in_len = ftell(in) / sizeof(int32_t);
    fseek(in, 0, SEEK_SET);

    uint8_t reverb_heap[RV_HEAP_SZ(FS, MAX_ROOM_SIZE)] = {0};
    reverb_room_t reverb = adsp_reverb_room_init(n_chans, fs,
                                                 max_room_size, room_size,
                                                 decay, damping, wet_gain,
                                                 dry_gain, pregain,
                                                 reverb_heap);

#if (PRINT_INIT != 0)
    printf("Wet (Q31): %ld\n", reverb.channel[0].wet_gain);
    printf("Dry (Q31): %ld\n", reverb.channel[0].dry_gain);
    printf("Pre (Q31): %ld\n", reverb.channel[0].pre_gain);
    printf("Comb Lens: ");
    for (int i = 0; i < N_COMBS; i++)
    {
        printf("%lu, ", reverb.channel[0].comb_lengths[i]);
    }
    printf("\n");
    printf("AP Lens: ");
    for (int i = 0; i < N_APS; i++)
    {
        printf("%lu, ", reverb.channel[0].ap_lengths[i]);
    }
    printf("\n");
    for (int i = 0; i < N_COMBS; i++)
    {
        printf("Comb %d:\n", i);
        printf("\tMax Delay: %lu\n", reverb.channel[0].combs[i].max_delay);
        printf("\tDelay: %lu\n", reverb.channel[0].combs[i].delay);
        printf("\tBuffer Idx: %lu\n", reverb.channel[0].combs[i].buffer_idx);
        printf("\tFB (Q31): %ld\n", reverb.channel[0].combs[i].feedback);
        printf("\tDamp 1 (Q31): %ld\n", reverb.channel[0].combs[i].damp_1);
        printf("\tDamp 2 (Q31): %ld\n", reverb.channel[0].combs[i].damp_2);
    }
    for (int i = 0; i < N_APS; i++)
    {
        printf("AP %d:\n", i);
        printf("\tMax Delay: %lu\n", reverb.channel[0].allpasses[i].max_delay);
        printf("\tDelay: %lu\n", reverb.channel[0].allpasses[i].delay);
        printf("\tFB (Q31): %ld\n", reverb.channel[0].allpasses[i].feedback);
    }
#endif

    for (int i = 0; i < in_len; i++)
    {
        int32_t samp = 0, samp_out = 0;
        fread(&samp, sizeof(int32_t), 1, in);
        samp_out = adsp_reverb_room(&reverb, samp, channel);
        fwrite(&samp_out, sizeof(int32_t), 1, out);
    }

    fclose(in);
    fclose(out);

    return 0;
}
