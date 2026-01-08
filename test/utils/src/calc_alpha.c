// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "control/helpers.h"

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
  FILE * in = _fopen("test_vector.bin", "rb");
  FILE * out = _fopen("out_vector.bin", "wb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(float);
  printf("inlen %d", in_len);
  fseek(in, 0, SEEK_SET);

  for (unsigned i = 0; i < in_len; i++)
  {
    float samp = 0;
    int32_t samp_out = 0;
    fread(&samp, sizeof(float), 1, in);
    //printf("%ld ", samp);
    samp_out = calc_alpha(48000.0, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
