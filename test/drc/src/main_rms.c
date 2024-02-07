#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <xcore/hwtimer.h>
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
  hwtimer_t tmr = hwtimer_alloc();

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  float th, at_al, re_al;

  fread(&th, sizeof(float), 1, lim_info);
  fread(&at_al, sizeof(float), 1, lim_info);
  fread(&re_al, sizeof(float), 1, lim_info);
  fclose(lim_info);

  limiter_t lim = (limiter_t){
              (env_detector_t){at_al, re_al, 0}, th, 1};

  //printf("%f %f %f\n", th, at_al, re_al);

  uint32_t begin, end;
  uint64_t acc = 0;
  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    begin = hwtimer_get_time(tmr);
    samp_out = adsp_limiter_rms(&lim, samp);
    end = hwtimer_get_time(tmr);
    if (end > begin) {
    acc += end - begin;
    } else {
      acc += UINT32_MAX - (end - begin);
    }
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }
  printf("average %f", (float)acc / (float)in_len);

  fclose(in);
  fclose(out);

  return 0;
}
