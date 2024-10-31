// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "control/reverb.h"
#include "control/reverb_plate.h"

FILE * _fopen(char * fname, char* mode) {
  FILE * fp = fopen(fname, mode);
  if (fp == NULL)
  {
    printf("Error opening a file %s\n", fname);
    exit(1);
  }
  return fp;
}

int main(int argc, char* argv[])
{
  int n_inputs = 1;
  FILE * in = _fopen("test_vector.bin", "rb");
  FILE * out = _fopen("out_vector.bin", "wb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / (n_inputs*sizeof(float));
  //printf("inlen %d", in_len);
  fseek(in, 0, SEEK_SET);

  for (unsigned i = 0; i < in_len; i++)
  {
    float samp = 0;
    fread(&samp, sizeof(float), 1, in);

#if defined(FLOAT2INT)
    int32_t ival = adsp_reverb_float2int(samp);
    fwrite(&ival, sizeof(int32_t), 1, out);
#elif defined(DB2INT)
    int32_t ival = adsp_reverb_db2int(samp);
    fwrite(&ival, sizeof(int32_t), 1, out);
#elif defined(DECAY2FEEDBACK)
    int32_t ival = adsp_reverb_calculate_feedback(samp);
    fwrite(&ival, sizeof(int32_t), 1, out);
#elif defined(CALCULATE_DAMPING)
    int32_t ival = adsp_reverb_calculate_damping(samp);
    fwrite(&ival, sizeof(int32_t), 1, out);
#elif defined(WET_DRY_MIX)
    int32_t gains[2];
    adsp_reverb_wet_dry_mix(gains, samp);
    fwrite(gains, sizeof(int32_t), 2, out);
#elif defined(WET_DRY_MIX_ST)
    int32_t gains[3];
    adsp_reverb_st_wet_dry_mix(gains, samp, 1.0);
    fwrite(gains, sizeof(int32_t), 3, out);
#elif defined(CUTOFF)
    int32_t ival = adsp_reverb_plate_calc_bandwidth(samp, 48000.0f);
    fwrite(&ival, sizeof(int32_t), 1, out);
#else
#error "config not defined"
#endif
    //printf("returned: %ld\n", ival);

    //fwrite(&ival, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
