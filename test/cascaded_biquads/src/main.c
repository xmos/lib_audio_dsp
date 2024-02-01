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
  int32_t DWORD_ALIGNED taps_buf[40] = {0};
  int32_t state[64] = {0};
  left_shift_t lsh[8] = {0};

  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * coeffs = _fopen("coeffs.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  for (unsigned n = 0; n < 8; n++){
    fread(&taps_buf[n * 5], sizeof(int32_t), 5, coeffs);
    fread(&lsh[n], sizeof(int32_t), 1, coeffs);
    //printf("%ld %ld %ld %ld %ld %d\n", taps_buf[n * 5], taps_buf[n * 5 + 1], taps_buf[n * 5 + 2], taps_buf[n * 5 + 3], taps_buf[n * 5 + 4], lsh[n]);
  }
  fclose(coeffs);

  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_cascaded_biquads_8b(samp, taps_buf, state, lsh);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
