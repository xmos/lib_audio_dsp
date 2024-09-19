// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"

int32_t adsp_cascaded_biquads_8b(int32_t new_sample,
                                q2_30 coeffs[40],
                                int32_t state[64],
                                left_shift_t lsh[8]
) {
  int32_t out = new_sample;
  for(unsigned n = 0; n < 8; n++)
  {
    out = adsp_biquad(out, &coeffs[5 * n], &state[8 * n], lsh[n]);
  }
  return out;
}
