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
  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * in1 = _fopen("../sig1_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);


  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp1 = 0, samp_out = 0;
    int64_t acc = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    fread(&samp1, sizeof(int32_t), 1, in1);
    //printf("%ld ", samp);
    acc += samp;
    acc += samp1;
    samp_out = adsp_saturate_32b(acc);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(in1);
  fclose(out);

  return 0;
}
