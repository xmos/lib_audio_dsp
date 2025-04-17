// Copyright 2024-2025 XMOS LIMITED.
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

int main()
{
  int32_t gains[4] = {0};
  int32_t slew = 0;

  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * in1 = _fopen("../sig1_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * gain = _fopen("gain.bin", "rb");


  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  fread(&slew, sizeof(int32_t), 1, gain);
  fread(gains, sizeof(int32_t), 4, gain);
  fclose(gain);

  printf("gains: %ld %ld\n", gains[0], gains[1]);

  crossfader_slew_t cfs = {.gain_1.gain=gains[0], .gain_1.target_gain=gains[2], .gain_1.slew_shift=slew,
                           .gain_2.gain=gains[1], .gain_2.target_gain=gains[3], .gain_2.slew_shift=slew,
                           .mix=0};

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp1 = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    fread(&samp1, sizeof(int32_t), 1, in1);
    //printf("%ld ", samp);
    samp_out = adsp_crossfader_slew(&cfs, samp, samp1);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }


  fclose(in);
  fclose(in1);
  fclose(out);

  return 0;
}
