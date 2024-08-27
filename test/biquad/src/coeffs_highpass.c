// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "control/biquad.h"

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
  int in_len = ftell(in) / (3*sizeof(float));
  printf("inlen %d", in_len);
  fseek(in, 0, SEEK_SET);

  for (unsigned i = 0; i < in_len; i++)
  {
    float samp[3] = {0};
    // int32_t samp_out = 0;
    fread(&samp, 3*sizeof(float), 1, in);

    float fc = samp[0];
    float fs = samp[1];
    float fq = samp[2];

    q2_30 coeffs[5] = {0};

    //printf("%ld ", samp);
    adsp_design_biquad_highpass(coeffs, fc, fs, fq);
    //printf("%ld ", samp_out);
    fwrite(&coeffs, 5*sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
