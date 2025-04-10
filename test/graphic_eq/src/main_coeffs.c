// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "control/helpers.h"

FILE * _fopen(char * fname, char* mode) {
  FILE * fp = fopen(fname, mode);
  if (fp == NULL)
  {
    printf("Error opening a file\n");
    exit(1);
  }
  return fp;
}

int main(int argc, char* argv[])
{
  float fs = atof(argv[1]);
  FILE * out = _fopen("coeffs_out.bin", "wb");

 printf("%f\n", fs);

  q2_30 * coeffs_buf = adsp_graphic_eq_10b_init(fs);

  for (int i=0; i < 50; i++){
    printf("%d: %ld\n", i, coeffs_buf[i]);
  }

  fwrite(&coeffs_buf[0], sizeof(int32_t), 50, out);
  fclose(out);
  return 0;
}
