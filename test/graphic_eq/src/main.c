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
  q2_30 coeffs_buf[50] = {0};
  int32_t DWORD_ALIGNED state[160] = {0};
  int32_t gains_buf[10] = {0};

  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * gains = _fopen("gains.bin", "rb");
  FILE * coeffs = _fopen("coeffs.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  fread(&gains_buf, sizeof(int32_t), 10, gains);
  fclose(gains);

  fread(&coeffs_buf, sizeof(int32_t), 50, coeffs);
  fclose(coeffs);

  // q2_30 * coeffs = adsp_graphic_eq_10b_init(48000.0f);

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_graphic_eq_10b(samp, gains_buf, coeffs_buf, state);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
