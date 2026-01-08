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

int main()
{
  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * info = _fopen("delay.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  uint32_t start_delay, max_delay;
  fread(&max_delay, sizeof(uint32_t), 1, info);
  fread(&start_delay, sizeof(uint32_t), 1, info);
  fclose(info);

  int32_t * buffer = (int32_t *) malloc(DELAY_DSP_REQUIRED_MEMORY_SAMPLES(max_delay));
  if (buffer == NULL)
  {
    printf("Error allocating memory\n");
    exit(2);
  }
  memset(buffer, 0, DELAY_DSP_REQUIRED_MEMORY_SAMPLES(max_delay));

  delay_t delay = (delay_t) {0, start_delay, max_delay, 0, buffer};

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_delay(&delay, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
