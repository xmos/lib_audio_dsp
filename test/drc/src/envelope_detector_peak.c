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
  FILE * env_info = _fopen("env_info.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  int32_t at_al, re_al;

  fread(&at_al, sizeof(int32_t), 1, env_info);
  fread(&re_al, sizeof(int32_t), 1, env_info);
  fclose(env_info);

  env_detector_t env_det = (env_detector_t){at_al, re_al, 0};

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    adsp_env_detector_peak(&env_det, samp);
    fwrite(&env_det.envelope, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
