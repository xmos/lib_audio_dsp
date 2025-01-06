#include "dsp/adsp.h"
#include "stdio.h"

int32_t adsp_biquad_slew_2(
  int32_t new_sample,
  q2_30 coeffs[8],
  q2_30 target_coeffs[8],
  int32_t state[8],
  left_shift_t lsh,
  int32_t slew_shift,
  bool print) {

    // for (int i=0; i < 5; i++){
    //     coeffs[i] += (target_coeffs[i] - coeffs[i]) >> slew_shift;
    // }

    if (print){
    printf("before_c: %ld\n", coeffs[0]);
    printf("before_t: %ld\n", target_coeffs[0]);
    printf("slew_shift: %ld\n", slew_shift);
  }

    int32_t DWORD_ALIGNED shift[8] = {slew_shift};
    asm volatile("vclrdr");
    asm volatile("ldc r11, 0x00");
    asm volatile("vsetc r11");

    register int32_t r11 asm("r11") = (int32_t)coeffs;
    // int32_t DWORD_ALIGNED tmp[8] = {0};

    // register int32_t r10 asm("r10") = (int32_t)coeffs;
    asm volatile("vldr %0[0]" :: "r" (r11));
    asm volatile("vlsub %0[0]" :: "r" (target_coeffs));
    // asm volatile("vstr %0[0]" :: "r" (coeffs));
    // asm volatile("vldr %0[0]" :: "r" (r11));
    
    // if (print){
    // asm volatile("vstr %0[0]" :: "r" (tmp));
    // asm volatile("vlashr %0[0], %0" :: "r" (tmp), "r" (slew_shift));
    asm volatile("vlsat %0[0]" :: "r" (shift));
    asm volatile("vladd %0[0]" :: "r" (coeffs));
    asm volatile("vstr %0[0]" :: "r" (coeffs));
    printf("after_add: %ld\n", coeffs[0]);
    // }
    // else{
    // asm volatile("vlsat %0[0]" :: "r" (shift));
    // // asm volatile("vstr r10[0]");
    // // asm volatile("vlashr r10[0], %0" :: "r" (slew_shift));
    // asm volatile("vladd %0[0]" :: "r" (coeffs));
    // asm volatile("vstr %0[0]" :: "r" (coeffs));
    // }
    return adsp_biquad(new_sample, coeffs, state, lsh);
  }