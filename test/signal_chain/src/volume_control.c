// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "dsp/adsp.h"

FILE * _fopen(char * fname, char* mode) {
  FILE * fp = fopen(fname, mode);
  if (fp == NULL)
  {
    printf("Error opening a file\n");
    exit(1);
  }
  return fp;
}

static inline void run_interval(FILE * in, FILE * out, unsigned bgn, unsigned end, volume_control_t * vol_ctl) {
  for (unsigned i = bgn; i < end; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_volume_control(vol_ctl, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }
}

int main()
{
  int32_t mute_test = 0;
  int32_t slew_shift = 0;
  int32_t gains[3] = {0};
  unsigned intervals[4] = {0};

  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * conf = _fopen("gain.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);
  intervals[1] = in_len / 3;
  intervals[2] = intervals[1] * 2;
  intervals[3] = in_len;

  fread(&mute_test, sizeof(int32_t), 1, conf);
  fread(&slew_shift, sizeof(int32_t), 1, conf);
  fread(gains, sizeof(int32_t), 3, conf);
  fclose(conf);

  //printf("ls %ld %ld %ld %ld\n", slew_shift, gains[0], gains[1], gains[2]);

  volume_control_t vol_ctl = (volume_control_t){gains[0], gains[0], slew_shift};

  run_interval(in, out, intervals[0], intervals[1], &vol_ctl);

  adsp_volume_control_set_gain(&vol_ctl, gains[1]);
  if (mute_test) {adsp_volume_control_mute(&vol_ctl);}

  run_interval(in, out, intervals[1], intervals[2], &vol_ctl);

  adsp_volume_control_set_gain(&vol_ctl, gains[2]);
  if (mute_test) {adsp_volume_control_unmute(&vol_ctl);}

  run_interval(in, out, intervals[2], intervals[3], &vol_ctl);

  fclose(in);
  fclose(out);

  return 0;
}
