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
  //FILE * in0 = _fopen("ch0.bin", "rb");
  //FILE * in1 = _fopen("ch1.bin", "rb");
  FILE * in = _fopen("../sig_2ch_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * comp_info = _fopen("info.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / (sizeof(int32_t) * 2); // two channels
  fseek(in, 0, SEEK_SET);

  int32_t th, at_al, re_al;
  float sl;

  fread(&th, sizeof(int32_t), 1, comp_info);
  fread(&at_al, sizeof(int32_t), 1, comp_info);
  fread(&re_al, sizeof(int32_t), 1, comp_info);
  fread(&sl, sizeof(float), 1, comp_info);
  fclose(comp_info);

  compressor_t comp = (compressor_t){
              (env_detector_t){at_al, re_al, 0}, th, 1 << 30, sl};

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp0 = 0, samp1 = 0, samp_out = 0;
    fread(&samp0, sizeof(int32_t), 1, in);
    fread(&samp1, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_compressor_rms_sidechain(&comp, samp0, samp1);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  //fclose(in1);
  fclose(out);

  return 0;
}
