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
  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * nse_info = _fopen("info.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  int32_t th, at_al, re_al;
  float slope;

  fread(&th, sizeof(int32_t), 1, nse_info);
  fread(&at_al, sizeof(int32_t), 1, nse_info);
  fread(&re_al, sizeof(int32_t), 1, nse_info);
  fread(&slope, sizeof(float), 1, nse_info);

  fclose(nse_info);
  if (!th) th = 1;
  noise_suppressor_expander_t nse = (noise_suppressor_expander_t){
              (env_detector_t){at_al, re_al, (1 << (Q_SIG)) - 1}, 0, 0, INT32_MAX, slope};
  adsp_noise_suppressor_expander_set_th(&nse, th);
  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_noise_suppressor_expander(&nse, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
