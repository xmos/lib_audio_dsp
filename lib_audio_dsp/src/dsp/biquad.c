#include "dsp/adsp.h"

int32_t adsp_biquad_slew(
  int32_t new_sample,
  q2_30 coeffs[5],
  q2_30 target_coeffs[5],
  int32_t state[8],
  left_shift_t lsh,
  int32_t slew_shift) {
    for (int i=0; i < 5; i++){
        coeffs[i] += (target_coeffs[i] - coeffs[i]) >> slew_shift;
    }
    return adsp_biquad(new_sample, coeffs, state, lsh);
  }