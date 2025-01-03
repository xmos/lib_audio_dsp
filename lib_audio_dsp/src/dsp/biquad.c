#include "dsp/adsp.h"

int32_t adsp_biquad_slew(
  int32_t new_sample,
  q2_30 coeffs[5],
  q2_30 target_coeffs[5],
  int32_t state[8],
  left_shift_t lsh,
  int32_t slew_shift) {

    // for (int i=0; i < 5; i++){
    //     coeffs[i] += (target_coeffs[i] - coeffs[i]) >> slew_shift;
    // }

    int32_t shift[8] = {slew_shift};
    asm volatile("vldc %0[0]" :: "r" (coeffs));
    asm volatile("vlsub %0[0]" :: "r" (target_coeffs));
    asm volatile("vlsat %0[0]" :: "r" (shift));
    // asm volatile("vlashr r11 %0" :: "r" (slew_shift));
    asm volatile("vladd %0[0]" :: "r" (coeffs));
    asm volatile("vstr %0[0]" :: "r" (coeffs));

    return adsp_biquad(new_sample, coeffs, state, lsh);
  }