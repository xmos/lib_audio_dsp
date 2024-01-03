#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "biquad.h"

filter_biquad_s32_t filter = {
  // Number of biquad sections in this filter block
  .biquad_count = 1,
  
  // Filter state, initialized to 0
  .state = {{0}},

  // Filter coefficients
  .coef = {{0}}
};

int main()
{
  int32_t taps_buf[5] = {0};
  FILE * in = fopen("sig_48k.bin", "rb");
  FILE * out = fopen("sig_out.bin", "wb");
  if ((in == NULL) || (out == NULL))
  {
    printf("Error opening a file\n");
    exit(1);
  }

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  float fs = 48000;
  float f = 1000;
  adsp_design_biquad_lowpass(taps_buf, f, fs, 0.7);
  //adsp_design_biquad_highpass(taps_buf, f, fs, 0.7);
  //adsp_design_biquad_notch(taps_buf, f, fs, 0.7);
  //adsp_design_biquad_allpass(taps_buf, f, fs, 0.7);
  //adsp_design_biquad_peaking(taps_buf, f, fs, 0.7, 3);
  printf("taps : b0 %ld b1 %ld b2 %ld a1 %ld a2 %ld\n", taps_buf[0], taps_buf[1], taps_buf[2], taps_buf[3], taps_buf[4]);

  for (unsigned i = 0; i < 5; i++)
  {
    //memset(&filter.coef[i][0], taps_buf[i], 8 * sizeof(int32_t));
    filter.coef[i][0] = taps_buf[i];
  }
  
  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samp = 0, samp_out = 0;
    fread(&samp, sizeof(int32_t), 1, in);
    //printf("%ld ", samp);
    //samp_out = adsp_apply_biquad(&filter, samp);
    samp_out = filter_biquad_s32(&filter, samp);
    //printf("%ld ", samp_out);
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  return 0;
}
