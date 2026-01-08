// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "dsp/adsp.h"
#include "control/adsp_control.h"

#define FS 48000
#define MAX_ROOM 1.0
#define PD_MS 1
#define PD_SAMPS (uint32_t)(PD_MS * FS / 1000)

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

  FILE *in = _fopen("../sig_2ch_48k.bin", "rb");
  FILE *out = _fopen("rv_sig_out.bin", "wb");
  FILE *info = _fopen("rv_info.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = (ftell(in) / sizeof(int32_t)) / 2; // stereo
  fseek(in, 0, SEEK_SET);

  int32_t pregain, wet1, wet2, dry, decay, dc_diff1, damp, dc_diff2, bw, in_diff1, in_diff2;

  fread(&pregain, sizeof(int32_t), 1, info);
  fread(&wet1, sizeof(int32_t), 1, info);
  fread(&wet2, sizeof(int32_t), 1, info);
  fread(&dry, sizeof(int32_t), 1, info);
  fread(&decay, sizeof(int32_t), 1, info);
  fread(&dc_diff2, sizeof(int32_t), 1, info);
  fread(&damp, sizeof(int32_t), 1, info);
  fread(&dc_diff1, sizeof(int32_t), 1, info);
  fread(&bw, sizeof(int32_t), 1, info);
  fread(&in_diff1, sizeof(int32_t), 1, info);
  fread(&in_diff2, sizeof(int32_t), 1, info);
  fclose(info);

  uint8_t reverb_heap[ADSP_RVP_HEAP_SZ(FS, PD_SAMPS)] = {0};
  reverb_plate_t rv;
  rv.pre_gain = pregain;
  rv.wet_gain1 = wet1;
  rv.wet_gain2 = wet2;
  rv.dry_gain = dry;
  rv.decay = decay;
  rv.lowpasses[0] = lowpass_1ord_init(bw);
  rv.lowpasses[1] = lowpass_1ord_init(damp);
  rv.lowpasses[2] = lowpass_1ord_init(damp);

  adsp_reverb_plate_init_filters(&rv, fs, dc_diff1, dc_diff2, in_diff1, in_diff2, PD_SAMPS, PD_SAMPS, reverb_heap);

  for (int i = 0; i < in_len; i++)
  {
    int32_t samp_l = 0, samp_r = 0, samp_out[2] = {0};
    fread(&samp_l, sizeof(int32_t), 1, in);
    fread(&samp_r, sizeof(int32_t), 1, in);
    adsp_reverb_plate(&rv, samp_out, samp_l, samp_r);
    fwrite(samp_out, sizeof(int32_t), 2, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
