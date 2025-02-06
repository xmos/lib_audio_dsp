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
  int32_t DWORD_ALIGNED taps_buf[8] = {0};
  // int32_t DWORD_ALIGNED taps_buf_1[8] = {0};
  int32_t DWORD_ALIGNED taps_buf_2[8] = {0};
  int32_t state[8] = {0};
  left_shift_t lsh = 0;
  left_shift_t lsh_2 = 0;
  int32_t shift = 0;

  FILE * in = _fopen("../slew_sig_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * coeffs = _fopen("coeffs.bin", "rb");
  FILE * coeffs_2 = _fopen("coeffs_2.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  fread(taps_buf, sizeof(int32_t), 5, coeffs);
  fread(&lsh, sizeof(int32_t), 1, coeffs);
  fread(&shift, sizeof(int32_t), 1, coeffs);
  fclose(coeffs);

  fread(taps_buf_2, sizeof(int32_t), 5, coeffs_2);
  fread(&lsh_2, sizeof(int32_t), 1, coeffs_2);
  fclose(coeffs_2);
  
  adsp_biquad_slew_state_t slew_state;
  adsp_biquad_slew_state_init(&slew_state, taps_buf, lsh, shift);

  int32_t * states[1] = {&state[0]};

  for (unsigned i = 0; i < in_len/2; i++)
  {
    adsp_biquad_slew_coeffs(&slew_state, states, 1);
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_biquad(samp, slew_state.coeffs, state, slew_state.lsh);
    // printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  adsp_biquad_slew_state_update_coeffs(&slew_state, taps_buf_2, lsh_2);

  for (unsigned i = in_len/2; i < in_len; i++)
  {
    adsp_biquad_slew_coeffs(&slew_state, states, 1);
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    samp_out = adsp_biquad(samp, slew_state.coeffs, state, slew_state.lsh);
    // printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
