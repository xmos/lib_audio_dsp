// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "xmath/filter.h"

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
  // this is just a simple wrapper for lib_xcore_math's filter_fir_s32,
  // for testing bit exactness against the Python model.

  right_shift_t rsh = 0;
  int32_t n_taps = 0;

  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * coeffs = _fopen("coeffs.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  fread(&rsh, sizeof(int32_t), 1, coeffs);
  fread(&n_taps, sizeof(int32_t), 1, coeffs);

  int32_t * taps_buf = calloc(sizeof(int32_t), n_taps);
  int32_t * samp_buf = calloc(sizeof(int32_t), n_taps);

  fread(taps_buf, sizeof(int32_t), n_taps, coeffs);

  //printf("%ld %ld %ld %ld %ld %d\n", taps_buf[0], taps_buf[1], taps_buf[2], taps_buf[3], taps_buf[4], lsh);
  fclose(coeffs);

  filter_fir_s32_t * filter = malloc(sizeof(filter_fir_s32_t));

  filter_fir_s32_init(filter, samp_buf, n_taps, taps_buf, rsh);

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = filter_fir_s32(filter, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
