// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "control/biquad.h"

FILE * _fopen(char * fname, char* mode) {
  FILE * fp = fopen(fname, mode);
  if (fp == NULL)
  {
    printf("Error opening a file\n");
    exit(1);
  }
  return fp;
}

int read_int();
float read_float();

enum bq_type {allpass, bandpass, bandstop, bypass, constq, gain, high_shelf,
              highpass, linkwitz, low_shelf, lowpass, mute, notch, peaking};

int main(int argc, char* argv[])
{
  enum bq_type this_bq = atoi(argv[1]);

  int n_inputs = 0;
  if (this_bq == allpass || this_bq == bandpass || this_bq == bandstop || this_bq == highpass || 
      this_bq == lowpass || this_bq == notch){
    n_inputs = 3;
  }
  else if (this_bq == constq || this_bq == high_shelf || this_bq == low_shelf || this_bq == peaking)
  {
    n_inputs = 4;
  }
  else if (this_bq == bypass || this_bq == gain || this_bq == mute)
  {
    n_inputs = 1;
  }
  else if (this_bq == linkwitz){
    n_inputs = 5;
  }
  else {
    printf("Unknown biquad type\n");
    exit(1);
    }
  FILE * in = _fopen("test_vector.bin", "rb");
  FILE * out = _fopen("out_vector.bin", "wb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / (n_inputs*sizeof(float));
  printf("inlen %d", in_len);
  fseek(in, 0, SEEK_SET);

  for (unsigned i = 0; i < in_len; i++)
  {
    float samp[5] = {0}; // never more than 4 inputs
    fread(&samp, sizeof(float), n_inputs, in);

    q2_30 coeffs[5] = {0};

    if (this_bq == allpass){
        adsp_design_biquad_allpass(coeffs, samp[0], samp[1], samp[2]);
    }
    else if (this_bq == bandpass){
        adsp_design_biquad_bandpass(coeffs, samp[0], samp[1], samp[2]);
    }
    else if (this_bq == bandstop){
        adsp_design_biquad_bandstop(coeffs, samp[0], samp[1], samp[2]);
    }
    else if (this_bq == bypass){
        adsp_design_biquad_bypass(coeffs);
    }
    else if (this_bq == constq){
        adsp_design_biquad_const_q(coeffs, samp[0], samp[1], samp[2], samp[3]);
    }
    else if (this_bq == gain){
        adsp_design_biquad_gain(coeffs, samp[0]);
    }
    else if (this_bq == high_shelf){
        adsp_design_biquad_highshelf(coeffs, samp[0], samp[1], samp[2], samp[3]);
    }
    else if (this_bq == highpass){
        adsp_design_biquad_highpass(coeffs, samp[0], samp[1], samp[2]);
    }
    else if (this_bq == linkwitz){
        adsp_design_biquad_linkwitz(coeffs, samp[0], samp[1], samp[2], samp[3], samp[4]);
    }
    else if (this_bq == lowpass){
        adsp_design_biquad_lowpass(coeffs, samp[0], samp[1], samp[2]);
    }
    else if (this_bq == low_shelf){
        adsp_design_biquad_lowshelf(coeffs, samp[0], samp[1], samp[2], samp[3]);
    }
    else if (this_bq == mute){
        adsp_design_biquad_mute(coeffs);
    }
    else if (this_bq == notch){
        adsp_design_biquad_notch(coeffs, samp[0], samp[1], samp[2]);
    }
    else if (this_bq == peaking){
        adsp_design_biquad_peaking(coeffs, samp[0], samp[1], samp[2], samp[3]);
    }
    else {
    printf("Unknown biquad type\n");
    exit(1);
    }

    fwrite(&coeffs, sizeof(int32_t), 5, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
