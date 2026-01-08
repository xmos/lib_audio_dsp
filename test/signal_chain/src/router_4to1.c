// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
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
  int32_t channel_states[4] = {1, 0, 0, 0}; // Default: only first channel active
  
  FILE * in = _fopen("../sig_48k.bin", "rb");
  FILE * in1 = _fopen("../sig1_48k.bin", "rb");
  FILE * out = _fopen("sig_out.bin", "wb");
  FILE * states_file = _fopen("channel_states.bin", "rb");

  fseek(in, 0, SEEK_END);
  int in_len = ftell(in) / sizeof(int32_t);
  fseek(in, 0, SEEK_SET);

  // Read channel states configuration
  int32_t channel_states_int[4];
  fread(channel_states_int, sizeof(int32_t), 4, states_file);
  fclose(states_file);
  
  // Convert to bool array
  for (int i = 0; i < 4; i++) {
    channel_states[i] = channel_states_int[i] != 0;
  }

  // Process the input signals
  for (unsigned i = 0; i < in_len; i++)
  {
    int32_t samples[4] = {0};
    int32_t samp_out = 0;
    
    // Read the input samples
    fread(&samples[0], sizeof(int32_t), 1, in);
    fread(&samples[1], sizeof(int32_t), 1, in1);
    // samples[2] and samples[3] duplicated from the first two inputs
    samples[2] = samples[0];
    samples[3] = samples[1];
    
    // Process through the router
    samp_out = adsp_router_4to1(channel_states, samples);
    
    // Write the output
    fwrite(&samp_out, sizeof(int32_t), 1, out);
  }

  fclose(in);
  fclose(in1);
  fclose(out);

  return 0;
}
