// Copyright 2024 XMOS LIMITED.
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
  int32_t DWORD_ALIGNED taps_buf[8] = {0};
  int32_t DWORD_ALIGNED taps_buf_1[8] = {0};
  int32_t DWORD_ALIGNED taps_buf_2[8] = {0};
  int32_t state[8] = {0};
  left_shift_t lsh = 0;
  int32_t shift = 0;

  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * coeffs = _fopen("coeffs.bin", "rb");
  FILE * coeffs_2 = _fopen("coeffs_2.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  fread(taps_buf, sizeof(int32_t), 5, coeffs);
  fread(&lsh, sizeof(int32_t), 1, coeffs);
  fclose(coeffs);

  coeffs = _fopen("coeffs.bin", "rb");
  fread(taps_buf_1, sizeof(int32_t), 5, coeffs);
  fclose(coeffs);

  fread(taps_buf_2, sizeof(int32_t), 5, coeffs_2);
  fread(&shift, sizeof(int32_t), 1, coeffs_2);
  fclose(coeffs_2);
  
  for (unsigned i = 0; i < in_len/2; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_biquad_slew(samp, taps_buf, taps_buf_1, state, lsh, shift, true);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  for (unsigned i = in_len/2; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_biquad_slew(samp, taps_buf, taps_buf_2, state, lsh, shift, true);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
