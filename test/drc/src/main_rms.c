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

  float_s32_t th;
  uq2_30 at_al, re_al;

  fread(&th.mant, sizeof(int32_t), 1, lim_info);
  fread(&th.exp, sizeof(exponent_t), 1, lim_info);
  fread(&at_al, sizeof(uq2_30), 1, lim_info);
  fread(&re_al, sizeof(uq2_30), 1, lim_info);

  limiter_t lim = (limiter_t){
                  (env_detector_t){at_al, re_al, (float_s32_t){0, SIG_EXP}},
                  th, (float_s32_t){0x40000000, -30}
  };

  //printf("%ld %d %ld %ld\n", th.mant, th.exp, at_al, re_al);

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_limiter_rms(&lim, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  return 0;
}
