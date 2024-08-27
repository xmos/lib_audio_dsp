// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include "control/biquad.h"

FILE * _fopen(char * fname, char* mode) {
  FILE * fp = fopen(fname, mode);
  if (fp == NULL)
  {
    printf("Error opening a file\n");
    printf("%s\n", strerror(errno));
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
  printf("inlen %d\n", in_len);
  fseek(in, 0, SEEK_SET);

  for (unsigned i = 0; i < in_len; i++)
  {
    float samp[3] = {0};
    // int32_t samp_out = 0;
    fread(&samp, sizeof(float), 3, in);

    float fc = samp[0];
    float fs = samp[1];
    float fq = samp[2];

    q2_30 coeffs[5] = {0};

    // printf("%d %f %f %f \n", i, samp[0], samp[1], samp[2]);
    // printf("New:\n");
    adsp_design_biquad_lowpass(coeffs, fc, fs, fq);

    // printf("%ld \n", coeffs[0]);
    fwrite(&coeffs, sizeof(int32_t), 5, out);
    // printf("%d %d\n", i, fwritten);
    // fflush(stdout);
    // fflush(out);
    // sleep(1);
  }

  fclose(in);
  fclose(out);

  return 0;
}
