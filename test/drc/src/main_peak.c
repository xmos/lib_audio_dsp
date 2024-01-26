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
  FILE * lim_info = _fopen("lim_info.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  float at, rt, th;
  
  fread(&th, sizeof(float), 1, lim_info);
  fread(&at, sizeof(float), 1, lim_info);
  fread(&rt, sizeof(float), 1, lim_info);
  //printf("%f %f %f\n", at, rt, th);
  
  limiter_t lim = adsp_limiter_peak_init(48000, th, at, rt);

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_limiter_peak(&lim, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  return 0;
}
